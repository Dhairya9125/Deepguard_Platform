"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";

const footerLinks = {
  Quick: [
    { label: "Home", href: "#hero" },
    { label: "Features", href: "#features" },
    { label: "Solutions", href: "#capabilities" },
    { label: "Pricing", href: "#pricing" },
    { label: "Contact", href: "#cta" },
  ],
  Product: [
    { label: "About", href: "#" },
    { label: "Changelog", href: "#" },
    { label: "Documentation", href: "#" },
    { label: "API Reference", href: "#" },
    { label: "Status", href: "#" },
  ],
  Social: [
    { label: "Twitter / X", href: "#" },
    { label: "GitHub", href: "#" },
    { label: "Discord", href: "#" },
    { label: "LinkedIn", href: "#" },
  ],
};

export default function Footer() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <footer className="relative border-t border-white/[0.06] bg-[#050816]">
      <div className="max-w-7xl mx-auto px-6 md:px-12 pt-20 pb-8">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="grid grid-cols-2 md:grid-cols-4 gap-10 md:gap-16"
        >
          <div className="col-span-2 md:col-span-1">
            <h3 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-white to-blue-300 bg-clip-text text-transparent mb-4">
              TRUX
            </h3>
            <p className="text-sm text-white/40 leading-relaxed max-w-xs">
              Enterprise-grade deepfake detection platform. Protect your organization from synthetic media threats.
            </p>
          </div>

          {Object.entries(footerLinks).map(([section, links]) => (
            <div key={section}>
              <h4 className="text-xs tracking-[0.2em] uppercase text-white/30 font-semibold mb-4">
                {section}
              </h4>
              <ul className="space-y-2.5">
                {links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      className="text-sm text-white/50 hover:text-white transition-colors duration-300"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-16 pt-8 border-t border-white/[0.06] flex flex-col md:flex-row items-center justify-between gap-4"
        >
          <p className="text-xs text-white/30">
            &copy; 2025 TRUX Inc. All rights reserved.
          </p>
          <div className="flex gap-6">
            <a href="#" className="text-xs text-white/30 hover:text-white/60 transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="text-xs text-white/30 hover:text-white/60 transition-colors">
              Terms of Service
            </a>
            <a href="#" className="text-xs text-white/30 hover:text-white/60 transition-colors">
              Cookies
            </a>
          </div>
        </motion.div>
      </div>
    </footer>
  );
}
