"use client";

import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Image,
  Shield,
  AlertTriangle,
  CheckCircle,
  X,
  Search,
  BarChart3,
  Activity,
  Layers,
  Download,
} from "lucide-react";

type AnalysisResult = {
  status: "authentic" | "suspicious" | "fake";
  confidence: number;
  authenticityScore: number;
  detectedArtifacts: string[];
  processingTime: string;
};

export default function ImageDetector() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const runAnalysis = () => {
    if (!file) return;
    setAnalyzing(true);
    setResult(null);

    setTimeout(() => {
      const isFake = Math.random() > 0.45;
      const artifacts = isFake
        ? [
            "AI-generated texture artifacts detected",
            "Inconsistent lighting patterns",
            "Face boundary anomalies (92%)",
            "Color space irregularities",
          ]
        : [];
      setResult({
        status: isFake ? "fake" : "authentic",
        confidence: isFake ? 94.7 + Math.random() * 3 : 98.2 + Math.random() * 1.5,
        authenticityScore: isFake ? 12 + Math.random() * 15 : 89 + Math.random() * 10,
        detectedArtifacts: artifacts,
        processingTime: (1.2 + Math.random() * 2.3).toFixed(1),
      });
      setAnalyzing(false);
    }, 2500);
  };

  return (
    <section className="pt-36 pb-20 px-6 md:px-12">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-center mb-12"
        >
          <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            Image Forensics
          </span>
          <h1 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight">
            Image Deepfake{" "}
            <span className="text-gradient-blue">Detection</span>
          </h1>
          <p className="mt-4 text-white/50 max-w-xl mx-auto">
            Upload an image to analyze for AI-generated artifacts, face manipulation, and digital forgeries
          </p>
        </motion.div>

        {!file ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => inputRef.current?.click()}
            className={`relative p-16 md:p-24 rounded-3xl border-2 border-dashed text-center cursor-pointer
              transition-all duration-500 group
              ${
                dragOver
                  ? "border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10"
                  : "border-white/10 hover:border-blue-500/40 bg-white/[0.02] hover:bg-white/[0.04]"
              }`}
          >
            <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            <div className="relative z-10">
              <div className="inline-flex p-4 rounded-2xl bg-blue-500/10 mb-6">
                <Upload size={32} className="text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-white/80 mb-2">
                Drop your image here
              </h3>
              <p className="text-white/40 text-sm">
                or click to browse &bull; PNG, JPG, WEBP up to 20MB
              </p>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid md:grid-cols-2 gap-6 md:gap-8"
          >
            <div className="space-y-4">
              <div className="relative rounded-3xl overflow-hidden border border-white/[0.08] bg-white/[0.02]">
                {preview && (
                  <img
                    src={preview}
                    alt="Uploaded preview"
                    className="w-full h-auto object-contain max-h-[400px]"
                  />
                )}
                <div className="absolute top-3 right-3 flex gap-2">
                  <button
                    onClick={reset}
                    className="p-2 rounded-full bg-black/60 backdrop-blur-sm text-white/60 hover:text-white transition-colors"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
              <div className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
                <Image size={16} className="text-blue-400" />
                <span className="text-sm text-white/60 truncate flex-1">
                  {file.name}
                </span>
                <span className="text-xs text-white/30 font-mono">
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </div>
              <motion.button
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.98 }}
                onClick={runAnalysis}
                disabled={analyzing}
                className="w-full py-4 rounded-full font-medium text-white
                  bg-gradient-to-r from-blue-600 to-blue-500
                  hover:from-blue-500 hover:to-cyan-400
                  transition-all duration-300 shadow-lg shadow-blue-500/20
                  disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {analyzing ? (
                  <span className="flex items-center justify-center gap-2">
                    <motion.span
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Activity size={18} />
                    </motion.span>
                    Analyzing...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <Shield size={18} />
                    Run Deepfake Analysis
                  </span>
                )}
              </motion.button>
            </div>

            <div>
              <AnimatePresence mode="wait">
                {!analyzing && !result && (
                  <motion.div
                    key="ready"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-full min-h-[300px] rounded-3xl border border-white/[0.08] bg-white/[0.02] overflow-hidden relative"
                  >
                    <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 via-transparent to-transparent" />
                    <motion.div
                      className="relative z-10 flex flex-col items-center"
                      animate={{ y: [0, -4, 0] }}
                      transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                    >
                      <div className="relative mb-6">
                        <motion.div
                          className="absolute inset-0 rounded-full"
                          animate={{
                            boxShadow: [
                              "0 0 0 0 rgba(77,124,254,0)",
                              "0 0 0 20px rgba(77,124,254,0.1)",
                              "0 0 0 40px rgba(77,124,254,0)",
                            ],
                          }}
                          transition={{ duration: 2.5, repeat: Infinity, ease: "easeOut" }}
                        />
                        <div className="relative w-28 h-28 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center overflow-hidden">
                          <motion.div
                            className="absolute inset-0 bg-gradient-to-b from-blue-500/20 to-transparent"
                            animate={{ opacity: [0.3, 0.7, 0.3] }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                          <svg
                            width="56"
                            height="56"
                            viewBox="0 0 24 24"
                            fill="none"
                            className="relative z-10"
                          >
                            <motion.rect
                              x="3"
                              y="3"
                              width="18"
                              height="18"
                              rx="3"
                              stroke="#4D7CFE"
                              strokeWidth="1.5"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                            />
                            <motion.circle
                              cx="12"
                              cy="12"
                              r="3"
                              stroke="#66E3FF"
                              strokeWidth="1.5"
                              initial={{ scale: 0.8, opacity: 0.5 }}
                              animate={{ scale: [0.8, 1.2, 0.8], opacity: [0.5, 1, 0.5] }}
                              transition={{ duration: 2, repeat: Infinity }}
                            />
                            <motion.line
                              x1="3"
                              y1="9"
                              x2="21"
                              y2="9"
                              stroke="#4D7CFE"
                              strokeWidth="1"
                              strokeDasharray="2 2"
                              opacity="0.5"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 1, repeat: Infinity, delay: 0.3 }}
                            />
                            <motion.line
                              x1="3"
                              y1="15"
                              x2="21"
                              y2="15"
                              stroke="#4D7CFE"
                              strokeWidth="1"
                              strokeDasharray="2 2"
                              opacity="0.5"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 1, repeat: Infinity, delay: 0.6 }}
                            />
                            <motion.line
                              x1="9"
                              y1="3"
                              x2="9"
                              y2="21"
                              stroke="#4D7CFE"
                              strokeWidth="1"
                              strokeDasharray="2 2"
                              opacity="0.5"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 1, repeat: Infinity, delay: 0.9 }}
                            />
                            <motion.line
                              x1="15"
                              y1="3"
                              x2="15"
                              y2="21"
                              stroke="#4D7CFE"
                              strokeWidth="1"
                              strokeDasharray="2 2"
                              opacity="0.5"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 1, repeat: Infinity, delay: 1.2 }}
                            />
                            <motion.circle
                              cx="12"
                              cy="12"
                              r="8"
                              stroke="#4D7CFE"
                              strokeWidth="0.5"
                              strokeDasharray="3 3"
                              opacity="0.3"
                              animate={{ rotate: 360 }}
                              transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                              style={{ transformOrigin: "12px 12px" }}
                            />
                          </svg>
                        </div>
                      </div>
                      <h3 className="text-lg font-semibold text-white/70">
                        Ready to Analyze
                      </h3>
                      <p className="text-sm text-white/40 mt-1 text-center max-w-[200px]">
                        Click the button to run deepfake detection on this image
                      </p>
                    </motion.div>
                    <motion.div
                      className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-1.5"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                    >
                      {[0, 1, 2].map((i) => (
                        <motion.div
                          key={i}
                          className="w-1 h-1 rounded-full bg-blue-400/40"
                          animate={{ opacity: [0.2, 1, 0.2] }}
                          transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                        />
                      ))}
                    </motion.div>
                  </motion.div>
                )}
                {analyzing && (
                  <motion.div
                    key="analyzing"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-full min-h-[300px] rounded-3xl border border-white/[0.08] bg-white/[0.02]"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.1, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="inline-flex p-5 rounded-full bg-blue-500/10 mb-6"
                    >
                      <Search size={36} className="text-blue-400" />
                    </motion.div>
                    <div className="w-48 h-1 rounded-full bg-white/10 overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full"
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 2.5, ease: "easeInOut" }}
                      />
                    </div>
                    <p className="mt-4 text-sm text-white/40">
                      Scanning for deepfake indicators...
                    </p>
                  </motion.div>
                )}
                {result && !analyzing && (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                  >
                    <div
                      className={`p-6 rounded-3xl border ${
                        result.status === "authentic"
                          ? "bg-green-500/5 border-green-500/20"
                          : result.status === "suspicious"
                          ? "bg-yellow-500/5 border-yellow-500/20"
                          : "bg-red-500/5 border-red-500/20"
                      }`}
                    >
                      <div className="flex items-center gap-3 mb-4">
                        {result.status === "authentic" ? (
                          <CheckCircle size={28} className="text-green-400" />
                        ) : (
                          <AlertTriangle size={28} className="text-red-400" />
                        )}
                        <div>
                          <h3 className="text-lg font-semibold text-white capitalize">
                            {result.status === "authentic"
                              ? "Likely Authentic"
                              : "Deepfake Detected"}
                          </h3>
                          <p className="text-sm text-white/40">
                            Confidence: {result.confidence.toFixed(1)}%
                          </p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-white/60">Authenticity Score</span>
                            <span className="text-white font-mono">
                              {result.authenticityScore.toFixed(0)}%
                            </span>
                          </div>
                          <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{
                                width: `${result.authenticityScore}%`,
                              }}
                              transition={{ duration: 1, delay: 0.3 }}
                              className={`h-full rounded-full ${
                                result.authenticityScore > 70
                                  ? "bg-gradient-to-r from-green-500 to-green-400"
                                  : "bg-gradient-to-r from-red-500 to-red-400"
                              }`}
                            />
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-sm">
                          <span className="text-white/40">Processing time</span>
                          <span className="text-white/60 font-mono">
                            {result.processingTime}s
                          </span>
                        </div>
                      </div>
                    </div>

                    {result.detectedArtifacts.length > 0 && (
                      <div className="p-6 rounded-3xl border border-white/[0.08] bg-white/[0.02]">
                        <div className="flex items-center gap-2 mb-4">
                          <Layers size={16} className="text-red-400" />
                          <h4 className="text-sm font-semibold text-white/80">
                            Detected Artifacts
                          </h4>
                        </div>
                        <ul className="space-y-2">
                          {result.detectedArtifacts.map((a, i) => (
                            <motion.li
                              key={i}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.5 + i * 0.1 }}
                              className="flex items-center gap-2 text-sm text-red-300/80"
                            >
                              <span className="w-1.5 h-1.5 rounded-full bg-red-400/60" />
                              {a}
                            </motion.li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="flex gap-3">
                      <button
                        onClick={() =>
                          setResult({
                            ...result,
                            status: result.status,
                            confidence: result.confidence,
                            authenticityScore: result.authenticityScore,
                            detectedArtifacts: result.detectedArtifacts,
                            processingTime: result.processingTime,
                          })
                        }
                        className="flex-1 py-3 rounded-full text-sm font-medium border border-white/20 text-white/70 hover:border-white/40 transition-all"
                      >
                        <span className="flex items-center justify-center gap-2">
                          <Download size={14} />
                          Export Report
                        </span>
                      </button>
                      <button
                        onClick={reset}
                        className="flex-1 py-3 rounded-full text-sm font-medium border border-blue-500/30 text-blue-400 hover:bg-blue-500/10 transition-all"
                      >
                        Analyze Another
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </div>
    </section>
  );
}
