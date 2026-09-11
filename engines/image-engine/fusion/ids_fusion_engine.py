"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.fusion.ids_fusion_engine
Layer   : Layer 3 — Fusion Engine

Combines multi-modal embeddings (Spatial, Frequency, Discrepancy, Noise, Fingerprint)
using Cross-Attention. Produces fake probabilities, OOD scores, and spatial heatmaps.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IDSFusionEngine(nn.Module):
    """
    Layer 3: Fusion Engine for Image Deepfake Detection.
    
    Accepts a dictionary of 512-d embeddings from the 5 Layer 2 branches.
    Returns:
        1. fake_probability: scalar [0, 1]
        2. manipulation_heatmap: spatial map (B, 1, 224, 224)
        3. OOD_score: scalar (AutoEncoder reconstruction MSE)
        4. artifact_embedding: The fused dense representation
    """
    def __init__(
        self,
        embed_dim: int = 512,
        num_branches: int = 5,
        num_heads: int = 8,
        transformer_layers: int = 2
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_branches = num_branches
        
        logger.info(f"Initialising IDSFusionEngine | embed_dim={embed_dim} | num_branches={num_branches}")
        
        # 1. Multi-Head Attention Fusion
        # Adds learnable positional encodings so the model knows WHICH branch is which
        self.branch_pos_embedding = nn.Parameter(torch.randn(1, num_branches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 2. Fake Probability Classification Head
        # Takes the flattened fused sequence
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * num_branches, embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # 3. OOD Score AutoEncoder
        # Compresses the fused embedding and reconstructs it. High loss = Out of Distribution.
        self.ood_encoder = nn.Sequential(
            nn.Linear(embed_dim * num_branches, 128),
            nn.GELU()
        )
        self.ood_decoder = nn.Sequential(
            nn.Linear(128, embed_dim * num_branches)
        )
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # 4. Manipulation Heatmap Decoder
        # Projects the 1D fused vector back into a 2D spatial heatmap (224x224)
        self.heatmap_decoder = nn.Sequential(
            # Start by projecting to a small 7x7 spatial map with 256 channels
            nn.Linear(embed_dim * num_branches, 256 * 7 * 7),
            nn.Unflatten(1, (256, 7, 7)), # (B, 256, 7, 7)
            
            # Upsample 1: 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Upsample 2: 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Upsample 3: 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # Upsample 4: 56x56 -> 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            
            # Upsample 5: 112x112 -> 224x224 (1 channel)
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, branch_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            branch_embeddings: Dict containing tensors of shape (B, 512).
                Keys expected: 'spatial', 'frequency', 'discrepancy', 'noise', 'fingerprint'
        """
        # Ensure we have the correct branches and order them consistently
        expected_keys = ['spatial', 'frequency', 'discrepancy', 'noise', 'fingerprint']
        
        # Stack embeddings into sequence (B, 5, 512)
        sequence = []
        for key in expected_keys:
            if key not in branch_embeddings:
                raise ValueError(f"Missing required branch embedding: {key}")
            sequence.append(branch_embeddings[key].unsqueeze(1))
            
        stacked = torch.cat(sequence, dim=1) # (B, 5, 512)
        
        # Add positional encoding so attention knows which branch is which
        stacked = stacked + self.branch_pos_embedding
        
        # 1. Multi-Head Attention Fusion
        fused_sequence = self.transformer_fusion(stacked) # (B, 5, 512)
        
        # Flatten for downstream heads
        fused_flat = fused_sequence.view(fused_sequence.size(0), -1) # (B, 2560)
        
        # 2. Artifact Embedding
        artifact_embedding = fused_flat
        
        # 3. Fake Probability
        fake_probability = self.classifier(fused_flat) # (B, 1)
        
        # 4. OOD Score
        encoded = self.ood_encoder(fused_flat)
        reconstructed = self.ood_decoder(encoded)
        # Compute MSE across features for each item in batch -> (B, 1)
        ood_score = self.mse_loss(reconstructed, fused_flat).mean(dim=1, keepdim=True)
        
        # 5. Manipulation Heatmap
        manipulation_heatmap = self.heatmap_decoder(fused_flat) # (B, 1, 224, 224)
        
        return {
            "fake_probability": fake_probability,
            "manipulation_heatmap": manipulation_heatmap,
            "OOD_score": ood_score,
            "artifact_embedding": artifact_embedding
        }
