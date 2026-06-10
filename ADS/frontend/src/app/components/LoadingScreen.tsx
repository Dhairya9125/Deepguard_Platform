"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTruxStore } from "../lib/store";

export default function LoadingScreen() {
  const { loading, loadingProgress, setLoadingProgress, setLoading } =
    useTruxStore();
  const frameRef = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<{ x: number; y: number; vx: number; vy: number; life: number }[]>([]);

  useEffect(() => {
    const duration = 2200;
    const start = performance.now();

    function tick(now: number) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setLoadingProgress(Math.round(eased * 100));
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(tick);
      } else {
        setTimeout(() => setLoading(false), 400);
      }
    }
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [setLoadingProgress, setLoading]);

  useEffect(() => {
    if (!loading) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = (canvas.width = window.innerWidth);
    let h = (canvas.height = window.innerHeight);
    const handleResize = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);

    const particles = particlesRef.current;
    const maxP = 30;

    let frame = 0;
    function draw() {
      if (!ctx) return;
      frame++;
      ctx.clearRect(0, 0, w, h);

      // Subtle radial grid
      ctx.strokeStyle = "rgba(77,124,254,0.03)";
      ctx.lineWidth = 0.5;
      const cx = w / 2;
      const cy = h / 2;
      for (let r = 50; r < Math.max(w, h) * 0.6; r += 50) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }
      const angles = 8;
      for (let i = 0; i < angles; i++) {
        const a = (i / angles) * Math.PI * 2 + frame * 0.001;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(a) * Math.max(w, h) * 0.6, cy + Math.sin(a) * Math.max(w, h) * 0.6);
        ctx.stroke();
      }

      // Particles
      if (particles.length < maxP && Math.random() < 0.12) {
        const angle = Math.random() * Math.PI * 2;
        const dist = 60 + Math.random() * 100;
        particles.push({
          x: cx + Math.cos(angle) * dist,
          y: cy + Math.sin(angle) * dist,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          life: 1,
        });
      }
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 0.006;
        if (p.life <= 0) {
          particles.splice(i, 1);
          continue;
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(102,227,255,${p.life * 0.5})`;
        ctx.fill();
      }

      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [loading]);

  return (
    <AnimatePresence>
      {loading && (
        <motion.div
          className="fixed inset-0 z-[9999] flex flex-col items-center justify-center overflow-hidden"
          style={{
            background: "radial-gradient(ellipse at center, #0A1628 0%, #050816 70%)",
          }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.8, ease: [0.76, 0, 0.24, 1] }}
        >
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
          />

          {/* Glow orbs */}
          <div className="absolute top-1/4 left-1/5 w-96 h-96 bg-blue-500/6 rounded-full blur-[150px] pointer-events-none" />
          <div className="absolute bottom-1/4 right-1/5 w-72 h-72 bg-cyan-400/4 rounded-full blur-[120px] pointer-events-none" />

          {/* Outer ring */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
          >
            <svg width="360" height="360" viewBox="0 0 360 360" className="absolute">
              <defs>
                <linearGradient id="outerRingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#4D7CFE" stopOpacity="0.15" />
                  <stop offset="50%" stopColor="#66E3FF" stopOpacity="0.05" />
                  <stop offset="100%" stopColor="#4D7CFE" stopOpacity="0" />
                </linearGradient>
              </defs>
              <motion.circle
                cx="180" cy="180" r="140"
                fill="none"
                stroke="url(#outerRingGrad)"
                strokeWidth="0.5"
                initial={{ rotate: 0 }}
                animate={{ rotate: 360 }}
                transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
              />
              <motion.circle
                cx="180" cy="180" r="120"
                fill="none"
                stroke="rgba(77,124,254,0.04)"
                strokeWidth="0.5"
                strokeDasharray="4 8"
                initial={{ rotate: 360 }}
                animate={{ rotate: 0 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              />
            </svg>
          </motion.div>

          {/* Central content */}
          <div className="relative z-10 text-center">
            <motion.div
              className="relative inline-flex items-baseline"
              animate={{ filter: ["brightness(1)", "brightness(1.1)", "brightness(1)"] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            >
              <h1 className="text-7xl md:text-9xl font-extrabold tracking-tight flex">
                <span className="bg-gradient-to-r from-white via-blue-200 to-cyan-200 bg-clip-text text-transparent">
                  TRU
                </span>
                <motion.span
                  className="text-gradient-blue inline-block"
                  initial={{ scale: 8, filter: "blur(10px) brightness(1.4)", opacity: 0 }}
                  animate={{
                    scale: [8, 1.04, 1],
                    filter: [
                      "blur(10px) brightness(1.4)",
                      "blur(1.5px) brightness(1.05)",
                      "blur(0px) brightness(1)",
                    ],
                    opacity: [0, 1, 1],
                  }}
                  transition={{
                    duration: 1.1,
                    times: [0, 0.55, 1],
                    ease: [0.22, 1, 0.36, 1],
                  }}
                >
                  X
                </motion.span>
              </h1>

              {/* Depth light bloom */}
              <motion.div
                className="absolute inset-0 flex items-center justify-center pointer-events-none"
                animate={{
                  opacity: [0, 0.25, 0],
                  scale: [0.3, 1.5, 2.5],
                }}
                transition={{ duration: 0.9, times: [0, 0.4, 1], ease: "easeOut", delay: 0.1 }}
              >
                <div className="w-48 h-48 rounded-full bg-gradient-to-r from-blue-500/0 via-cyan-300/10 to-transparent blur-3xl" />
              </motion.div>
            </motion.div>

            {/* Underline glow */}
            <motion.div
              className="mx-auto mt-2 h-[2px] rounded-full"
              style={{
                width: "0%",
                background: "linear-gradient(90deg, transparent, rgba(77,124,254,0.4), rgba(102,227,255,0.6), transparent)",
              }}
              animate={{ width: ["0%", "100%", "0%"] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
            />

            <motion.p
              className="mt-6 text-sm md:text-base text-white/25 tracking-[0.4em] uppercase font-light"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              Deepfake Detection Platform
            </motion.p>
          </div>

          {/* Progress section */}
          <motion.div
            className="relative z-10 mt-20 w-60"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <div className="relative h-[2px] bg-white/5 overflow-hidden rounded-full">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{
                  width: `${loadingProgress}%`,
                  background: "linear-gradient(90deg, #4D7CFE, #66E3FF, #4D7CFE)",
                  backgroundSize: "200% 100%",
                }}
                animate={{
                  backgroundPosition: ["0% 0%", "200% 0%"],
                }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              />
              <motion.div
                className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-cyan-300 blur-[4px]"
                style={{ left: `calc(${loadingProgress}% - 5px)` }}
              />
            </div>
            <div className="mt-3 flex justify-between items-center">
              <span className="text-[10px] text-white/15 font-mono tracking-[0.25em]">
                INITIALIZING
              </span>
              <span className="text-xs text-white/35 font-mono tabular-nums tracking-wider">
                {loadingProgress}%
              </span>
            </div>
          </motion.div>

          {/* Status dots */}
          <motion.div
            className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <div className="flex items-center gap-3">
              <span className="text-[10px] text-white/15 tracking-[0.2em] uppercase">
                Loading
              </span>
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className="w-1 h-1 rounded-full"
                  style={{
                    background: loadingProgress > 30 + i * 20 ? "#66E3FF" : "rgba(255,255,255,0.08)",
                    boxShadow: loadingProgress > 30 + i * 20 ? "0 0 6px rgba(102,227,255,0.4)" : "none",
                  }}
                  animate={loadingProgress <= 30 + i * 20 ? { opacity: [0.2, 0.6, 0.2] } : {}}
                  transition={{ duration: 1, repeat: Infinity, delay: i * 0.25 }}
                />
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
