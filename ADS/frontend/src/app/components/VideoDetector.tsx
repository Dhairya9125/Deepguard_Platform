"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { submitVideo } from "../lib/api";
import {
  Upload,
  Shield,
  AlertTriangle,
  X,
  Film,
  Search,
  Activity,
} from "lucide-react";

export default function VideoDetector() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollingStatus, setPollingStatus] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("video/")) return;
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
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
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setError(null);
    setPollingStatus(null);
  };

  const runAnalysis = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    setPollingStatus("Uploading video...");

    try {
      const { job_id } = await submitVideo(file);
      router.push(`/results/${job_id}`);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Unexpected error during analysis.";
      setError(message);
      setAnalyzing(false);
      setPollingStatus(null);
    }
  };

  return (
    <section className="pt-28 pb-20 px-6 md:px-12">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="flex flex-col items-center text-center mb-12"
        >
          <span className="inline-block mb-4 text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            Video Forensics
          </span>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tight mb-6">
            Video Deepfake <span className="text-gradient-blue">Detection</span>
          </h1>
          <p className="text-white/50 max-w-xl mx-auto leading-relaxed">
            Upload a video to analyze for AI-generated lip-syncing, face-swapping, and temporal anomalies
          </p>
        </motion.div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 rounded-2xl border border-red-500/30 bg-red-500/10 text-red-300 text-sm flex items-start gap-3"
          >
            <AlertTriangle size={18} className="mt-0.5 flex-shrink-0 text-red-400" />
            <div>
              <p className="font-semibold text-red-300 mb-0.5">Analysis Failed</p>
              <p className="text-red-300/70">{error}</p>
              <p className="mt-1 text-red-300/50 text-xs">
                Make sure the VDS backend is running: <code className="font-mono">uvicorn apps.api.main:app --port 8001</code>
              </p>
            </div>
          </motion.div>
        )}

        {analyzing && pollingStatus && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-6 p-3 rounded-xl border border-purple-500/20 bg-purple-500/5 text-purple-300/70 text-xs font-mono flex items-center gap-2"
          >
            <motion.span
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="w-1.5 h-1.5 rounded-full bg-purple-400 flex-shrink-0"
            />
            {pollingStatus}
          </motion.div>
        )}

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
                  ? "border-purple-500 bg-purple-500/10 shadow-lg shadow-purple-500/10"
                  : "border-white/10 hover:border-purple-500/40 bg-white/[0.02] hover:bg-white/[0.04]"
              }`}
          >
            <div className="absolute inset-0 bg-gradient-to-b from-purple-500/5 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            <div className="relative z-10">
              <div className="inline-flex p-4 rounded-2xl bg-purple-500/10 mb-6">
                <Upload size={32} className="text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-white/80 mb-2">
                Drop your video here
              </h3>
              <p className="text-white/40 text-sm">
                or click to browse &bull; MP4, MOV, WebM up to 200MB
              </p>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
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
                  <video
                    src={preview}
                    controls
                    className="w-full h-auto max-h-[320px] object-contain"
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
                <Film size={16} className="text-purple-400" />
                <span className="text-sm text-white/60 truncate flex-1">
                  {file.name}
                </span>
                <span className="text-xs text-white/30 font-mono">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB
                </span>
              </div>
              <motion.button
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.98 }}
                onClick={runAnalysis}
                disabled={analyzing}
                className="w-full py-4 rounded-full font-medium text-white
                  bg-gradient-to-r from-purple-600 to-blue-500
                  hover:from-purple-500 hover:to-blue-400
                  transition-all duration-300 shadow-lg shadow-purple-500/20
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
                    Analyzing frames...
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
                {!analyzing && (
                  <motion.div
                    key="ready"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-full min-h-[300px] rounded-3xl border border-white/[0.08] bg-white/[0.02] overflow-hidden relative"
                  >
                    <div className="absolute inset-0 bg-gradient-to-b from-purple-500/5 via-transparent to-transparent" />
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
                              "0 0 0 0 rgba(168,85,247,0)",
                              "0 0 0 20px rgba(168,85,247,0.1)",
                              "0 0 0 40px rgba(168,85,247,0)",
                            ],
                          }}
                          transition={{ duration: 2.5, repeat: Infinity, ease: "easeOut" }}
                        />
                        <div className="relative w-28 h-28 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center overflow-hidden">
                          <motion.div
                            className="absolute inset-0 bg-gradient-to-b from-purple-500/20 to-transparent"
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
                              x="2"
                              y="4"
                              width="20"
                              height="16"
                              rx="2"
                              stroke="#A855F7"
                              strokeWidth="1.5"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                            />
                            <motion.circle
                              cx="12"
                              cy="12"
                              r="2.5"
                              fill="#A855F7"
                              fillOpacity="0.4"
                              animate={{ r: [2.5, 3.5, 2.5], fillOpacity: [0.2, 0.6, 0.2] }}
                              transition={{ duration: 2, repeat: Infinity }}
                            />
                            <motion.rect
                              x="6"
                              y="8"
                              width="3"
                              height="2"
                              rx="0.5"
                              fill="#A855F7"
                              fillOpacity="0.5"
                              animate={{ opacity: [0.3, 0.7, 0.3] }}
                              transition={{ duration: 1.5, repeat: Infinity }}
                            />
                            <motion.rect
                              x="6"
                              y="12"
                              width="3"
                              height="2"
                              rx="0.5"
                              fill="#A855F7"
                              fillOpacity="0.5"
                              animate={{ opacity: [0.3, 0.7, 0.3] }}
                              transition={{ duration: 1.5, repeat: Infinity, delay: 0.3 }}
                            />
                            <motion.rect
                              x="6"
                              y="16"
                              width="3"
                              height="2"
                              rx="0.5"
                              fill="#A855F7"
                              fillOpacity="0.5"
                              animate={{ opacity: [0.3, 0.7, 0.3] }}
                              transition={{ duration: 1.5, repeat: Infinity, delay: 0.6 }}
                            />
                            <motion.rect
                              x="15"
                              y="8"
                              width="3"
                              height="2"
                              rx="0.5"
                              fill="#4D7CFE"
                              fillOpacity="0.5"
                              animate={{ opacity: [0.3, 0.7, 0.3] }}
                              transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
                            />
                            <motion.rect
                              x="15"
                              y="12"
                              width="3"
                              height="2"
                              rx="0.5"
                              fill="#4D7CFE"
                              fillOpacity="0.5"
                              animate={{ opacity: [0.3, 0.7, 0.3] }}
                              transition={{ duration: 1.5, repeat: Infinity, delay: 0.7 }}
                            />
                            <motion.rect
                              x="15"
                              y="16"
                              width="3"
                              height="2"
                              rx="0.5"
                              fill="#4D7CFE"
                              fillOpacity="0.5"
                              animate={{ opacity: [0.3, 0.7, 0.3] }}
                              transition={{ duration: 1.5, repeat: Infinity, delay: 1.0 }}
                            />
                            <motion.path
                              d="M12 7v2m0 3v2m0 3v2"
                              stroke="#A855F7"
                              strokeWidth="0.75"
                              strokeDasharray="2 2"
                              opacity="0.3"
                              animate={{ opacity: [0.2, 0.5, 0.2] }}
                              transition={{ duration: 2, repeat: Infinity }}
                            />
                            <motion.circle
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="#A855F7"
                              strokeWidth="0.5"
                              strokeDasharray="6 4"
                              opacity="0.3"
                              animate={{ rotate: 360 }}
                              transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                              style={{ transformOrigin: "12px 12px" }}
                            />
                          </svg>
                        </div>
                      </div>
                      <h3 className="text-lg font-semibold text-white/70">
                        Ready to Analyze
                      </h3>
                      <p className="text-sm text-white/40 mt-1 text-center max-w-[200px]">
                        Click the button to run frame-by-frame deepfake detection
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
                          className="w-1 h-1 rounded-full bg-purple-400/40"
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
                      className="inline-flex p-5 rounded-full bg-purple-500/10 mb-6"
                    >
                      <Search size={36} className="text-purple-400" />
                    </motion.div>
                    <div className="w-48 h-1 rounded-full bg-white/10 overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-purple-500 to-blue-400 rounded-full"
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 4, ease: "easeInOut" }}
                      />
                    </div>
                    <p className="mt-4 text-sm text-white/40">
                      Submitting video...
                    </p>
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
