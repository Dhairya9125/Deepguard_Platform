"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";

export default function AboutAI() {
  const sectionRef = useRef<HTMLElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: "-100px" });

  return (
    <section
      id="about"
      ref={sectionRef}
      className="relative w-full py-28 md:py-40 overflow-hidden"
    >
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <div className="grid md:grid-cols-2 gap-12 md:gap-20 items-start">
          <motion.div
            initial={{ opacity: 0, x: -60 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          >
            <span className="inline-block text-xs tracking-[0.3em] uppercase text-white/40 mb-6 font-mono">
              About TRUX
            </span>
            <h2 className="text-[clamp(28px,4vw,44px)] font-extrabold leading-[1.15] tracking-[-0.02em]">
              TRUX is an AI-powered forensic intelligence platform that detects deepfakes across image, audio, and video with unparalleled accuracy.
            </h2>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 60 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.15, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="space-y-6 text-base md:text-lg leading-relaxed text-white/60"
          >
            <p>
              Our advanced neural networks analyze digital media for subtle artifacts, inconsistencies, and manipulation patterns that human eyes can&rsquo;t see.
            </p>
            <p>
              Built on proprietary deep learning models trained on millions of samples, TRUX provides enterprise-grade protection against the growing threat of synthetic media.
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
