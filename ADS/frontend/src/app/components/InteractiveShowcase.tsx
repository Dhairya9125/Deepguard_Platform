"use client";

import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

const showcases = [
  {
    title: "Image Forensics Lab",
    desc: "Upload an image and watch as our AI scans every pixel for manipulation artifacts, generative fingerprints, and editing inconsistencies in real time.",
    side: "left" as const,
  },
  {
    title: "Audio Deepfake Detector",
    desc: "Analyze audio recordings for synthetic voice, splicing, and acoustic anomalies. Visualize waveform patterns and spectral signatures of manipulation.",
    side: "right" as const,
  },
  {
    title: "Video Integrity Scanner",
    desc: "Frame-by-frame video analysis that detects deepfake faces, lip-sync manipulation, and temporal inconsistencies across entire recordings.",
    side: "left" as const,
  },
];

const Visual = ({
  index,
  inView,
}: {
  index: number;
  inView: boolean;
}) => {
  const [particles] = useState<{x: number[], y: number[], duration: number, delay: number}[]>(() =>
    [...Array(6)].map(() => ({
      x: [Math.random() * 200 - 100, Math.random() * 200 - 100],
      y: [Math.random() * 200 - 100, Math.random() * 200 - 100],
      duration: 3 + Math.random() * 2,
      delay: Math.random() * 2,
    }))
  );
  const [waveform] = useState<{h: number, duration: number}[]>(() =>
    [4, 6, 8, 12, 18, 24, 28, 24, 18, 12, 8, 6, 4, 6, 8, 12, 18, 24, 28, 24, 18, 12, 8, 6, 4].map((h) => ({
      h,
      duration: 0.8 + Math.random() * 0.4,
    }))
  );

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={inView ? { opacity: 1, scale: 1 } : {}}
      transition={{
        duration: 0.8,
        delay: 0.3,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      className="relative flex items-center justify-center min-h-[180px]"
    >
      {index === 0 && (
        <div className="relative w-full h-full flex items-center justify-center">
          {/* Scanning frame */}
          <motion.div
            className="relative w-48 h-48 md:w-56 md:h-56"
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
          >
            <div className="absolute top-0 left-0 w-10 h-10 border-t-2 border-l-2 border-blue-400/60 rounded-tl-lg" />
            <div className="absolute top-0 right-0 w-10 h-10 border-t-2 border-r-2 border-blue-400/60 rounded-tr-lg" />
            <div className="absolute bottom-0 left-0 w-10 h-10 border-b-2 border-l-2 border-blue-400/60 rounded-bl-lg" />
            <div className="absolute bottom-0 right-0 w-10 h-10 border-b-2 border-r-2 border-blue-400/60 rounded-br-lg" />

            {/* Grid overlay */}
            <svg className="absolute inset-4 w-[calc(100%-32px)] h-[calc(100%-32px)] opacity-20" viewBox="0 0 100 100">
              <line x1="33" y1="0" x2="33" y2="100" stroke="#4D7CFE" strokeWidth="0.5" />
              <line x1="66" y1="0" x2="66" y2="100" stroke="#4D7CFE" strokeWidth="0.5" />
              <line x1="0" y1="33" x2="100" y2="33" stroke="#4D7CFE" strokeWidth="0.5" />
              <line x1="0" y1="66" x2="100" y2="66" stroke="#4D7CFE" strokeWidth="0.5" />
            </svg>

            {/* Face silhouette */}
            <svg className="absolute inset-6 w-[calc(100%-48px)] h-[calc(100%-48px)] opacity-30" viewBox="0 0 80 100" fill="none">
              <ellipse cx="40" cy="45" rx="30" ry="38" stroke="#66E3FF" strokeWidth="1" />
              <circle cx="28" cy="35" r="3" fill="#66E3FF" />
              <circle cx="52" cy="35" r="3" fill="#66E3FF" />
              <path d="M30 55 Q40 62 50 55" stroke="#66E3FF" strokeWidth="1" fill="none" />
            </svg>

            {/* Scanning beam */}
            <motion.div
              className="absolute left-4 right-4 h-[2px] bg-gradient-to-r from-transparent via-blue-400 to-transparent"
              animate={{ top: ["10%", "85%", "10%"] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            />
          </motion.div>

          {/* Data particles */}
          {particles.map((p, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-blue-400/60"
              animate={{
                x: p.x,
                y: p.y,
                opacity: [0, 1, 0],
              }}
              transition={{
                duration: p.duration,
                repeat: Infinity,
                delay: p.delay,
                ease: "linear",
              }}
            />
          ))}
        </div>
      )}

      {index === 1 && (
        <div className="relative w-full h-full flex items-center justify-center">
          {/* Waveform visualization */}
          <motion.div
            className="flex items-end gap-1.5 h-32"
            animate={{ scale: [1, 1.02, 1] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          >
            {waveform.map(
              (w, i) => (
                <motion.div
                  key={i}
                  className="w-1.5 rounded-full"
                  style={{
                    background: `linear-gradient(to top, rgba(77,124,254,0.4), rgba(102,227,255,0.8))`,
                  }}
                  animate={{
                    height: [
                      `${w.h + 10}%`,
                      `${w.h + 30}%`,
                      `${w.h + 10}%`,
                    ],
                  }}
                  transition={{
                    duration: w.duration,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                />
              )
            )}
          </motion.div>

          {/* Spectral spread lines */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            animate={{ opacity: [0.1, 0.3, 0.1] }}
            transition={{ duration: 4, repeat: Infinity }}
          >
            <svg viewBox="0 0 200 200" className="w-full h-full opacity-20">
              {[...Array(5)].map((_, i) => (
                <motion.circle
                  key={i}
                  cx="100"
                  cy="100"
                  r={20 + i * 15}
                  stroke="#66E3FF"
                  strokeWidth="0.5"
                  fill="none"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{
                    duration: 2 + i * 0.3,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                />
              ))}
            </svg>
          </motion.div>
        </div>
      )}

      {index === 2 && (
        <div className="relative w-full h-full flex items-center justify-center">
          {/* Film strip */}
          <motion.div
            className="relative flex gap-2"
            animate={{ y: [0, -4, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          >
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="relative w-16 h-20 md:w-20 md:h-24 rounded-lg overflow-hidden border border-white/10"
                initial={{ opacity: 0.3 }}
                animate={{
                  opacity: [0.3, 1, 0.3],
                  scale: [0.95, 1, 0.95],
                }}
                transition={{
                  duration: 2.5,
                  repeat: Infinity,
                  delay: i * 0.3,
                  ease: "easeInOut",
                }}
              >
                {/* Frame content - simple shape */}
                <div className="absolute inset-1 rounded bg-white/5 flex items-center justify-center">
                  <div className="w-8 h-8 md:w-10 md:h-10 rounded border border-purple-400/30 flex items-center justify-center">
                    <motion.div
                      className="w-4 h-4 rounded-full bg-purple-400/20"
                      animate={{ scale: [1, 1.3, 1] }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: i * 0.3,
                      }}
                    />
                  </div>
                </div>
                {/* Film perforations */}
                <div className="absolute left-0.5 top-1 bottom-1 flex flex-col justify-between">
                  {[...Array(4)].map((_, j) => (
                    <div key={j} className="w-1.5 h-1.5 rounded-full bg-white/20" />
                  ))}
                </div>
                <div className="absolute right-0.5 top-1 bottom-1 flex flex-col justify-between">
                  {[...Array(4)].map((_, j) => (
                    <div key={j} className="w-1.5 h-1.5 rounded-full bg-white/20" />
                  ))}
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* Play button overlay */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            animate={{ opacity: [0.2, 0.5, 0.2] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            <div className="w-14 h-14 md:w-16 md:h-16 rounded-full border border-purple-400/30 flex items-center justify-center backdrop-blur-sm bg-purple-500/5">
              <div className="w-0 h-0 border-t-[8px] border-t-transparent border-l-[14px] border-l-purple-400/70 border-b-[8px] border-b-transparent ml-1" />
            </div>
          </motion.div>
        </div>
      )}
    </motion.div>
  );
};
function ShowcaseRow({
  item,
  index,
  inView,
}: {
  item: (typeof showcases)[0];
  index: number;
  inView: boolean;
}) {
  const isLeft = item.side === "left";
  const Content = (
    <motion.div
      initial={{ opacity: 0, x: isLeft ? -40 : 40 }}
      animate={inView ? { opacity: 1, x: 0 } : {}}
      transition={{
        duration: 0.8,
        delay: 0.2,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      className="flex flex-col justify-center"
    >
      <span className="text-xs tracking-[0.3em] uppercase text-white/30 font-mono mb-4">
        {String(index + 1).padStart(2, "0")}
      </span>
      <h3 className="text-2xl md:text-3xl font-bold text-white mb-4">
        {item.title}
      </h3>
      <p className="text-white/50 leading-relaxed max-w-md">{item.desc}</p>
    </motion.div>
  );



  return (
    <div
      className={`grid md:grid-cols-2 gap-8 md:gap-16 items-center ${
        isLeft ? "" : "md:direction-rtl"
      }`}
      style={{ direction: isLeft ? "ltr" : "rtl" }}
    >
      <div style={{ direction: "ltr" }}>{isLeft ? Content : <Visual index={index} inView={inView} />}</div>
      <div style={{ direction: "ltr" }}>{isLeft ? <Visual index={index} inView={inView} /> : Content}</div>
    </div>
  );
}

export default function InteractiveShowcase() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="features" className="relative py-32 md:py-40 px-6 md:px-12 border-t border-white/[0.06]">
      <div className="max-w-7xl mx-auto">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 40 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="text-center mb-24"
        >
          <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            Showcase
          </span>
          <h2 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight">
            Platform in <span className="text-gradient-blue">Action</span>
          </h2>
        </motion.div>

        <div className="space-y-24 md:space-y-32">
          {showcases.map((item, i) => (
            <ShowcaseRow key={item.title} item={item} index={i} inView={inView} />
          ))}
        </div>
      </div>
    </section>
  );
}
