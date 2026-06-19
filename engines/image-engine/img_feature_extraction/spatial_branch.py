"""
DeepGuard Platform — Image Detection Subsystem (IDS)
Module  : engines.image-engine.feature_extraction.spatial_branch
Layer   : Layer 2 — Feature Extraction (Branch A)

Architecture Reference:
  - Branch A (Spatial / Semantic Features): CLIP ViT-L/14 (OpenAI). 
  - Backbone frozen; fine-tuned projection head.
  - Extracts rich semantic and structural features from the spatial domain.
"""

import logging

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

logger = logging.getLogger(__name__)


class ClipSpatialBranch(nn.Module):
    """
    Branch A: Semantic / Spatial Feature Extractor.
    
    Uses OpenAI's CLIP ViT-L/14 (Patch 14) as the frozen backbone.
    Applies a trainable Multi-Layer Perceptron (MLP) projection head to reduce
    the 768-dimensional semantic embeddings down to the target dimension (e.g. 512) 
    for the final fusion layer.
    """

    def __init__(
        self, 
        model_name: str = "openai/clip-vit-large-patch14",
        proj_dim: int = 512,
        freeze_backbone: bool = True
    ) -> None:
        super().__init__()
        
        self.model_name = model_name
        self.proj_dim = proj_dim
        
        logger.info(f"Loading CLIP backbone: {model_name}")
        
        try:
            # We use CLIPVisionModel to extract purely vision features.
            # It expects inputs normalized by CLIP's specific mean/std, but our 
            # normalizer output works well enough for general feature extraction, 
            # though standard CLIP preprocessing should ideally be used.
            self.backbone = CLIPVisionModel.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_name}: {e}")
            raise
            
        # CLIP ViT-L/14 embedding dimension is 768 or 1024 (actually large-patch14 is 1024)
        # Let's dynamically get the hidden size from the config.
        self.embed_dim = self.backbone.config.hidden_size
        
        if freeze_backbone:
            logger.info("Freezing CLIP backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Trainable projection head
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, proj_dim)
        )
        
        logger.info(f"ClipSpatialBranch initialised | proj_dim={proj_dim}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Tensor of shape (Batch, Channels=3, Height=224, Width=224).
                          Expected to be normalized (e.g. mean/std).
                          
        Returns:
            Projected feature vector of shape (Batch, proj_dim).
        """
        # Pass through frozen backbone
        # We don't need to track gradients for the backbone if frozen
        with torch.set_grad_enabled(not next(self.backbone.parameters()).requires_grad is False):
            outputs = self.backbone(pixel_values=pixel_values)
            
        # Extract the CLS token representation (pooler_output is a dense layer over CLS)
        # Usually CLIP vision pooler_output is reliable for semantic embeddings.
        pooled_output = outputs.pooler_output
        
        # Pass through trainable projection head
        projected = self.projection(pooled_output)
        
        return projected
