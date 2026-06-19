"use client";

import { useRef } from "react";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import MagneticButton from "./MagneticButton";

const Hero3D = dynamic(() => import("./Hero3D"), { ssr: false });

export default function Hero() {
  const sectionRef = useRef<HTMLElement>(null);

  return (
    <section
      id="hero"
      ref={sectionRef}
      className="relative min-h-screen flex items-center overflow-hidden"
    >
      <div className="absolute inset-0 z-0">
        <Hero3D />
      </div>

      <div
        className="absolute inset-0 z-[1] pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 30% 50%, transparent 30%, rgba(5,8,22,0.6) 60%, #050816 100%)",
        }}
      />

      <div className="relative z-10 w-full max-w-7xl mx-auto px-6 md:px-12 pt-32 pb-24">
        <div className="max-w-2xl flex flex-col gap-6">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
          >
            <span className="inline-block text-xs tracking-[0.3em] uppercase text-white/40 mb-2 font-mono">
              TRUX
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="text-[clamp(40px,7.5vw,96px)] font-extrabold leading-[1.0] md:leading-[0.92] tracking-tight break-words mb-4"
          >
            <span>
              <span className="text-gradient-blue">T</span>ruth{" "}
              <span className="text-gradient-blue">R</span>ecognition{" "}
              <span className="text-gradient-blue">U</span>nmasking{" "}
              <span className="text-gradient-blue">X</span>pose
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="text-base md:text-lg leading-relaxed text-white/60 max-w-lg"
          >
            Enterprise-grade deepfake detection across image, audio, and video. Our AI forensics platform helps organizations verify authenticity and combat disinformation.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="mt-10 flex flex-wrap gap-4"
          >
            <MagneticButton>
              <button
                onClick={() => {
                  document
                    .querySelector("#features")
                    ?.scrollIntoView({ behavior: "smooth" });
                }}
                className="group relative px-8 py-4 text-sm font-medium text-white rounded-full
                  bg-gradient-to-r from-blue-600 to-blue-500
                  hover:from-blue-500 hover:to-cyan-400
                  transition-all duration-500 shadow-lg shadow-blue-500/25
                  hover:shadow-blue-500/40 active:scale-[0.98]
                  overflow-hidden"
              >
                <span className="relative z-10 flex items-center gap-2">
                  Explore TRUX
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    className="group-hover:translate-x-1 transition-transform duration-300"
                  >
                    <path
                      d="M3 8h10M9 4l4 4-4 4"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </span>
                <span className="absolute inset-0 -translate-x-full group-hover:translate-x-0 transition-transform duration-700 bg-gradient-to-r from-transparent via-white/10 to-transparent skew-x-[-20deg]" />
              </button>
            </MagneticButton>

            <MagneticButton>
              <button
                onClick={() => {
                  document
                    .querySelector("#about")
                    ?.scrollIntoView({ behavior: "smooth" });
                }}
                className="px-8 py-4 text-sm font-medium text-white/80 rounded-full
                  border border-white/20 hover:border-white/40
                  transition-all duration-300
                  active:scale-[0.98]"
              >
                Learn More
              </button>
            </MagneticButton>
          </motion.div>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10"
      >
        <div className="flex flex-col items-center gap-2 text-white/30">
          <span className="text-[10px] tracking-[0.25em] uppercase font-mono">
            Scroll
          </span>
          <motion.div
            animate={{ y: [0, 6, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            className="w-[1px] h-8 bg-gradient-to-b from-white/40 to-transparent"
          />
        </div>
      </motion.div>
    </section>
  );
}
