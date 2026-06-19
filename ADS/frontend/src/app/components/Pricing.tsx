"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { Check } from "lucide-react";
import MagneticButton from "./MagneticButton";
import { useTruxStore } from "../lib/store";

const plans = [
  {
    name: "Starter",
    price: "$99",
    period: "/month",
    desc: "For small teams exploring deepfake detection.",
    features: [
      "1,000 media analyses/month",
      "Image + audio detection",
      "Email support",
      "Basic reports",
      "7-day history",
    ],
    cta: "Start Trial",
    highlighted: false,
  },
  {
    name: "Pro",
    price: "$499",
    period: "/month",
    desc: "For growing organizations with active detection needs.",
    features: [
      "25,000 media analyses/month",
      "Image + audio + video detection",
      "Priority support",
      "Full forensic reports",
      "90-day history",
      "API access",
      "Custom integrations",
    ],
    cta: "Start Free Trial",
    highlighted: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    desc: "Tailored solutions for large organizations.",
    features: [
      "Unlimited analysis",
      "All formats + real-time",
      "Dedicated support",
      "Custom SLAs",
      "White-label reports",
      "On-premise option",
      "Unlimited history",
      "Priority model updates",
    ],
    cta: "Contact Sales",
    highlighted: false,
  },
];

export default function Pricing() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });
  const { setModalOpen } = useTruxStore();

  return (
    <section id="pricing" className="relative py-32 md:py-40 px-6 md:px-12 border-t border-white/[0.06]">
      <div className="max-w-7xl mx-auto">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 40 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="text-center mb-20"
        >
          <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            Pricing
          </span>
          <h2 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight">
            Simple, <span className="text-gradient-blue">transparent</span> pricing
          </h2>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6 md:gap-8 max-w-5xl mx-auto items-start">
          {plans.map((plan, i) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 40 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{
                duration: 0.8,
                delay: 0.1 * i,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
              className={`relative p-8 md:p-10 rounded-3xl border transition-all duration-500
                ${
                  plan.highlighted
                    ? "bg-gradient-to-b from-blue-600/10 to-transparent border-blue-500/40 shadow-xl shadow-blue-500/10 scale-[1.02] md:scale-[1.05]"
                    : "bg-white/[0.03] border-white/[0.08] hover:border-white/20"
                }`}
            >
              {plan.highlighted && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full bg-gradient-to-r from-blue-600 to-cyan-500 text-[10px] font-bold text-white tracking-wider uppercase">
                  Most Popular
                </div>
              )}
              <h3 className="text-lg font-semibold text-white/60 mb-1">{plan.name}</h3>
              <div className="flex items-end gap-1 mb-2">
                <span className="text-4xl md:text-5xl font-extrabold text-white">
                  {plan.price}
                </span>
                {plan.period && (
                  <span className="text-white/40 text-sm mb-1.5">{plan.period}</span>
                )}
              </div>
              <p className="text-white/40 text-sm mb-8">{plan.desc}</p>
              <ul className="space-y-3 mb-8">
                {plan.features.map((f) => (
                  <li key={f} className="flex items-start gap-3 text-sm text-white/60">
                    <Check size={16} className="text-blue-400 mt-0.5 flex-shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
              <MagneticButton>
                <button
                  onClick={() => setModalOpen(true)}
                  className={`w-full py-3 text-sm font-medium rounded-full transition-all duration-300 active:scale-[0.98]
                    ${
                      plan.highlighted
                        ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40"
                        : "border border-white/20 text-white/80 hover:border-white/40"
                    }`}
                >
                  {plan.cta}
                </button>
              </MagneticButton>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
