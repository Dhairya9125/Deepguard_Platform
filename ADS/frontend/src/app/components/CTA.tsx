"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import MagneticButton from "./MagneticButton";
import { useTruxStore } from "../lib/store";

export default function CTA() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });
  const { setModalOpen } = useTruxStore();

  return (
    <section
      id="cta"
      className="relative py-32 md:py-48 px-6 md:px-12 overflow-hidden"
    >
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at center, rgba(77,124,254,0.08) 0%, transparent 60%)",
        }}
      />
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/4 w-64 h-64 bg-blue-500/5 rounded-full blur-[100px]" />
        <div className="absolute bottom-1/3 right-1/4 w-48 h-48 bg-cyan-400/5 rounded-full blur-[80px]" />
      </div>

      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 40 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="relative z-10 max-w-3xl mx-auto flex flex-col items-center text-center"
      >
        <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono mb-6">
          Get Started
        </span>
        <h2 className="text-4xl md:text-7xl font-extrabold tracking-tight leading-[1.05] mb-6">
          Ready to{" "}
          <span className="text-gradient-blue">protect the truth</span>?
        </h2>
        <p className="text-white/50 text-lg max-w-xl mx-auto leading-relaxed">
          Join thousands of organizations worldwide using TRUX to detect deepfakes and verify digital media authenticity.
        </p>
        <div className="mt-10 flex flex-wrap gap-4 justify-center">
          <MagneticButton>
            <button
              onClick={() => setModalOpen(true)}
              className="group relative px-8 py-4 text-sm font-medium text-white rounded-full
                bg-gradient-to-r from-blue-600 to-blue-500
                hover:from-blue-500 hover:to-cyan-400
                transition-all duration-500 shadow-lg shadow-blue-500/25
                hover:shadow-blue-500/40 active:scale-[0.98]
                overflow-hidden"
            >
              <span className="relative z-10">Get Started Free</span>
              <span className="absolute inset-0 -translate-x-full group-hover:translate-x-0 transition-transform duration-700 bg-gradient-to-r from-transparent via-white/10 to-transparent skew-x-[-20deg]" />
            </button>
          </MagneticButton>
          <MagneticButton>
            <button
              className="px-8 py-4 text-sm font-medium text-white/80 rounded-full
                border border-white/20 hover:border-white/40
                transition-all duration-300 active:scale-[0.98]"
            >
              Talk to Sales
            </button>
          </MagneticButton>
        </div>
      </motion.div>
    </section>
  );
}
