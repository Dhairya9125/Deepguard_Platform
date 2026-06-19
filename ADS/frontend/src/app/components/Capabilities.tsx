"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";

const capabilities = [
  {
    num: "01",
    title: "Visual Forensics",
    desc: "Deep analysis of image and video pixels for generative artifacts, compression inconsistencies, and AI-synthesized content.",
  },
  {
    num: "02",
    title: "Audio Analysis",
    desc: "Detection of synthetic voice, audio splicing, and acoustic anomalies using spectral analysis and waveform forensics.",
  },
  {
    num: "03",
    title: "Cross-Modal AI",
    desc: "Correlate findings across image, audio, and video simultaneously. Our cross-modal AI identifies inconsistencies that single-format analysis misses.",
  },
];

export default function Capabilities() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="capabilities" className="relative py-32 md:py-40 px-6 md:px-12">
      <div className="max-w-7xl mx-auto">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 40 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="flex flex-col items-center text-center mb-20"
        >
          <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            Capabilities
          </span>
          <h2 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight">
            <span className="text-gradient-blue">Forensic Capabilities</span>
          </h2>
          <p className="mt-4 text-white/50 max-w-xl mx-auto leading-relaxed">
            Powered by advanced neural architectures purpose-built for synthetic media detection.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6 md:gap-8">
          {capabilities.map((cap, i) => (
            <motion.div
              key={cap.num}
              initial={{ opacity: 0, y: 40 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{
                duration: 0.8,
                delay: 0.15 * i,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
              className="group relative p-8 md:p-10 rounded-3xl bg-white/[0.03] border border-white/[0.08]
                hover:bg-white/[0.06] hover:border-blue-500/20 transition-all duration-500"
            >
              <span className="text-5xl md:text-7xl font-black text-white/[0.04] absolute top-4 right-6 select-none">
                {cap.num}
              </span>
              <div className="relative z-10">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/10 border border-blue-500/20 flex items-center justify-center mb-6">
                  <div className="w-5 h-5 rounded-full bg-blue-400/60" />
                </div>
                <h3 className="text-xl md:text-2xl font-bold text-white mb-3">
                  {cap.title}
                </h3>
                <p className="text-white/50 leading-relaxed">{cap.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
