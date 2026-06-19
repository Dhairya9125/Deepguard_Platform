"use client";

import { useRef, useState, useEffect } from "react";
import { motion, useInView } from "framer-motion";

const stats = [
  { value: 99.97, suffix: "%", label: "Detection Accuracy", decimals: 2 },
  { value: 50, suffix: "M+", label: "Media Files Analyzed", decimals: 0, multiplier: 1_000_000 },
  { value: 12, suffix: "K+", label: "Enterprise Clients", decimals: 0, multiplier: 1_000 },
  { value: 99.99, suffix: "%", label: "Platform Uptime", decimals: 2 },
];

function AnimatedNumber({
  target,
  suffix,
  decimals,
  multiplier,
  inView,
}: {
  target: number;
  suffix: string;
  decimals: number;
  multiplier?: number;
  inView: boolean;
}) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!inView) return;
    const total = multiplier ? target * multiplier : target;
    const duration = 2000;
    const start = performance.now();

    function tick(now: number) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(eased * total);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [inView, target, multiplier]);

  const display = multiplier
    ? (count / multiplier).toFixed(decimals)
    : count.toFixed(decimals);

  return (
    <span>
      {display}
      {suffix}
    </span>
  );
}

export default function Statistics() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section className="relative py-24 md:py-32 px-6 md:px-12 border-t border-white/[0.06]">
      <div className="max-w-7xl mx-auto">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 40 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="text-center mb-16"
        >
          <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            By the Numbers
          </span>
          <h2 className="mt-4 text-4xl md:text-5xl font-extrabold tracking-tight">
            Trusted by <span className="text-gradient-blue">industry leaders</span>
          </h2>
        </motion.div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{
                duration: 0.6,
                delay: 0.1 * i,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
              className="group relative p-8 rounded-2xl bg-white/[0.03] border border-white/[0.08]
                hover:bg-white/[0.06] hover:border-blue-500/20 transition-all duration-500 text-center"
            >
              <div className="text-4xl md:text-5xl font-extrabold text-white mb-2 tabular-nums">
                <AnimatedNumber
                  target={stat.value}
                  suffix={stat.suffix}
                  decimals={stat.decimals}
                  multiplier={"multiplier" in stat ? stat.multiplier : undefined}
                  inView={inView}
                />
              </div>
              <p className="text-sm text-white/40">{stat.label}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
