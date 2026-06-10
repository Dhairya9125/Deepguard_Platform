"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { Cpu, Globe, Box, Zap } from "lucide-react";

const features = [
  {
    icon: Cpu,
    title: "Multi-Format Detection",
    description:
      "Analyze images, audio recordings, and video footage for deepfake manipulation across all major formats with a single unified platform.",
  },
  {
    icon: Globe,
    title: "Real-Time Analysis",
    description:
      "Get instant forensic results with our high-speed inference engine. Detect manipulated media in seconds, not hours.",
  },
  {
    icon: Box,
    title: "Forensic Evidence Reporting",
    description:
      "Generate comprehensive, court-admissible forensic reports with detailed artifact analysis and confidence scoring.",
  },
  {
    icon: Zap,
    title: "Continuous Model Updates",
    description:
      "Our AI models evolve constantly, staying ahead of the latest deepfake generation techniques and emerging threats.",
  },
];

const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.15,
    },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] as const },
  },
};

export default function FeatureGrid() {
  const sectionRef = useRef<HTMLElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: "-100px" });

  return (
    <section ref={sectionRef} className="relative w-full py-28 md:py-40 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate={isInView ? "visible" : "hidden"}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                variants={cardVariants}
                className="glass-card group p-8 md:p-10 cursor-default"
              >
                <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-6 group-hover:bg-primary-blue/10 transition-colors duration-300">
                  <Icon className="w-6 h-6 text-primary-blue group-hover:text-cyan-accent transition-colors duration-300" />
                </div>
                <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                <p className="text-white/60 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
}
