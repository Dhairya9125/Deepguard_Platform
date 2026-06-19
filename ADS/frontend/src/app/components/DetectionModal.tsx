"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { Image, Headphones, Video, ArrowRight, X } from "lucide-react";
import { useTruxStore } from "../lib/store";

const detectionTypes = [
  {
    id: "image",
    title: "Image Deepfake Detector",
    desc: "Analyze photos and images for AI-generated artifacts, face manipulation, and digital forgeries.",
    icon: Image,
    path: "/detect/image",
    gradient: "from-blue-600 to-blue-400",
  },
  {
    id: "audio",
    title: "Audio Deepfake Detector",
    desc: "Detect synthetic voice, audio splicing, and AI-generated speech with spectral analysis.",
    icon: Headphones,
    path: "/detect/audio",
    gradient: "from-cyan-500 to-blue-500",
  },
  {
    id: "video",
    title: "Video Deepfake Detector",
    desc: "Comprehensive video forensics including frame analysis, lip-sync inconsistencies, and temporal artifacts.",
    icon: Video,
    path: "/detect/video",
    gradient: "from-purple-600 to-blue-500",
  },
];

export default function DetectionModal() {
  const { isModalOpen, setModalOpen } = useTruxStore();
  const router = useRouter();

  const handleSelect = (path: string) => {
    setModalOpen(false);
    router.push(path);
  };

  return (
    <AnimatePresence>
      {isModalOpen && (
        <motion.div
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4 md:p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <motion.div
            className="absolute inset-0 bg-black/70 backdrop-blur-xl"
            onClick={() => setModalOpen(false)}
          />

          <motion.div
            className="relative z-10 max-w-4xl w-full"
            initial={{ opacity: 0, scale: 0.92, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 30 }}
            transition={{ type: "spring", damping: 28, stiffness: 200 }}
          >
            <button
              onClick={() => setModalOpen(false)}
              className="absolute -top-12 right-0 text-white/40 hover:text-white/80 transition-colors p-1"
              aria-label="Close"
            >
              <X size={20} />
            </button>

            <div className="text-center mb-10">
              <h2 className="text-3xl md:text-5xl font-extrabold tracking-tight">
                Choose Detection{" "}
                <span className="text-gradient-blue">Type</span>
              </h2>
              <p className="mt-3 text-white/50 text-sm md:text-base max-w-lg mx-auto">
                Select the media type you want to analyze for deepfake indicators
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-4 md:gap-6">
              {detectionTypes.map((type, i) => {
                const Icon = type.icon;
                return (
                  <motion.button
                    key={type.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * i, duration: 0.5 }}
                    className="group relative p-6 md:p-8 rounded-3xl text-left
                      bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08]
                      hover:bg-white/[0.06] hover:border-blue-500/30
                      transition-all duration-500 cursor-pointer
                      hover:shadow-xl hover:shadow-blue-500/5"
                    onClick={() => handleSelect(type.path)}
                  >
                    <div
                      className={`inline-flex p-3 rounded-2xl bg-gradient-to-br ${type.gradient} mb-5 shadow-lg`}
                    >
                      <Icon size={24} className="text-white" />
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">
                      {type.title}
                    </h3>
                    <p className="text-sm text-white/50 leading-relaxed">
                      {type.desc}
                    </p>
                    <div className="mt-5 flex items-center gap-2 text-sm text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <span>Analyze now</span>
                      <ArrowRight size={14} />
                    </div>
                  </motion.button>
                );
              })}
            </div>

            <div className="mt-10 text-center">
              <button
                onClick={() => setModalOpen(false)}
                className="text-sm text-white/40 hover:text-white/70 transition-colors tracking-wide"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
