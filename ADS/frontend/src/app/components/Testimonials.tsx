"use client";

import { useRef, useEffect, useState } from "react";
import { motion, useInView } from "framer-motion";

const testimonials = [
  {
    quote:
      "TRUX caught a sophisticated deepfake that bypassed our previous security measures. It's now an essential part of our verification pipeline.",
    author: "Marcus Rivera",
    title: "CISO at GlobalNews Network",
  },
  {
    quote:
      "The cross-modal analysis is a game-changer. We can now verify image, audio, and video in one seamless workflow.",
    author: "Sarah Chen",
    title: "Head of Digital Forensics at EuroPol",
  },
  {
    quote:
      "Accuracy and speed are unmatched. TRUX processes our entire media library in hours instead of weeks.",
    author: "James Adeyemi",
    title: "CTO of VerifyTrust",
  },
  {
    quote:
      "As a news agency fighting disinformation, TRUX gives us the confidence to authenticate content at scale.",
    author: "Elena Voss",
    title: "Director of Standards at WorldPress",
  },
  {
    quote:
      "The continuous model updates keep us ahead of the latest generation techniques. Essential for security operations.",
    author: "David Park",
    title: "VP Security at CyberShield Inc.",
  },
  {
    quote:
      "Deployment was seamless and the API integration took less than a day. Results were immediate and impressive.",
    author: "Aiko Tanaka",
    title: "Lead Engineer at TrustLab",
  },
];

export default function Testimonials() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !inView) return;

    let animationId: number;

    function scroll() {
      if (!isHovered && el) {
        el.scrollLeft += 0.5;
        if (el.scrollLeft >= el.scrollWidth / 2) {
          el.scrollLeft = 0;
        }
      }
      animationId = requestAnimationFrame(scroll);
    }
    animationId = requestAnimationFrame(scroll);
    return () => cancelAnimationFrame(animationId);
  }, [inView, isHovered]);

  return (
    <section className="relative py-32 md:py-40 px-6 md:px-12 border-t border-white/[0.06]">
      <div className="max-w-7xl mx-auto">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 40 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="text-center mb-16"
        >
          <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            Testimonials
          </span>
          <h2 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight">
            What our <span className="text-gradient-blue">clients</span> say
          </h2>
        </motion.div>
      </div>

      <div
        ref={scrollRef}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className="flex gap-6 overflow-x-auto scrollbar-hide px-6 md:px-12 pb-4"
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        {[...testimonials, ...testimonials].map((t, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.05 * (i % testimonials.length) }}
            className="flex-shrink-0 w-[340px] md:w-[400px] p-8 rounded-3xl
              bg-white/[0.04] border border-white/[0.08]
              hover:bg-white/[0.06] hover:border-blue-500/20
              transition-all duration-500"
          >
            <div className="flex mb-4">
              {[1, 2, 3, 4, 5].map((s) => (
                <svg
                  key={s}
                  width="14"
                  height="14"
                  viewBox="0 0 14 14"
                  className="text-blue-400 mr-0.5"
                  fill="currentColor"
                >
                  <path d="M7 0l2.15 4.36 4.81.7-3.48 3.4.82 4.8L7 10.5l-4.3 2.26.82-4.8L.04 5.06l4.81-.7L7 0z" />
                </svg>
              ))}
            </div>
            <p className="text-white/70 leading-relaxed mb-6 text-sm">
              &ldquo;{t.quote}&rdquo;
            </p>
            <div>
              <p className="text-white font-semibold text-sm">{t.author}</p>
              <p className="text-white/40 text-xs mt-0.5">{t.title}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
