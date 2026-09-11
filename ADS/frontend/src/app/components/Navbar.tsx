"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "../lib/utils";
import MagneticButton from "./MagneticButton";
import { useTruxStore } from "../lib/store";

const NAV_LINKS = [
  { label: "Home", href: "#hero" },
  { label: "Features", href: "#features" },
  { label: "Solutions", href: "#capabilities" },
  { label: "Pricing", href: "#pricing" },
  { label: "Contact", href: "#cta" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { setModalOpen } = useTruxStore();
  const pathname = usePathname();
  const router = useRouter();
  const isHome = pathname === "/";

  useEffect(() => {
    const handle = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handle, { passive: true });
    return () => window.removeEventListener("scroll", handle);
  }, []);

  useEffect(() => {
    if (isHome && window.location.hash) {
      setTimeout(() => {
        const el = document.querySelector(window.location.hash);
        if (el) el.scrollIntoView({ behavior: "smooth" });
      }, 300);
    }
  }, [isHome]);

  const scrollTo = useCallback(
    (href: string) => {
      setMobileOpen(false);
      if (isHome) {
        const el = document.querySelector(href);
        if (el) el.scrollIntoView({ behavior: "smooth" });
      } else {
        router.push(`/${href}`);
      }
    },
    [isHome, router]
  );

  return (
    <>
      <motion.nav
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={cn(
          "fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-center justify-between",
          "px-6 py-3 w-[calc(100%-32px)] max-w-6xl rounded-full",
          "backdrop-blur-2xl border transition-all duration-500",
          scrolled
            ? "bg-[#050816]/90 border-white/10 shadow-2xl shadow-blue-500/5"
            : "bg-white/5 border-white/8"
        )}
      >
        <a
          href="#hero"
          onClick={(e) => { e.preventDefault(); scrollTo("#hero"); }}
          className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white to-blue-300 bg-clip-text text-transparent"
        >
          TRUX
        </a>

        <div className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              onClick={(e) => { e.preventDefault(); scrollTo(link.href); }}
              className="text-sm text-white/60 hover:text-white transition-colors duration-300 tracking-wide"
            >
              {link.label}
            </a>
          ))}
        </div>

        <div className="hidden md:block">
          <MagneticButton>
            <button
              onClick={() => setModalOpen(true)}
              className="relative px-5 py-2.5 text-sm font-medium text-white rounded-full
                bg-gradient-to-r from-blue-600 to-blue-500
                hover:from-blue-500 hover:to-cyan-400
                transition-all duration-300 shadow-lg shadow-blue-500/25
                hover:shadow-blue-500/40 active:scale-[0.98]"
            >
              Start Now
            </button>
          </MagneticButton>
        </div>

        <button
          className="md:hidden text-white p-2"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </motion.nav>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, x: "100%" }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed inset-0 z-40 flex flex-col items-center justify-center gap-8
              bg-[#050816]/98 backdrop-blur-2xl md:hidden"
          >
            {NAV_LINKS.map((link, i) => (
              <motion.a
                key={link.label}
                href={link.href}
                onClick={(e) => { e.preventDefault(); scrollTo(link.href); }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className="text-3xl font-bold text-white/80 hover:text-white transition-colors"
              >
                {link.label}
              </motion.a>
            ))}
            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              onClick={() => setModalOpen(true)}
              className="mt-4 px-8 py-3 text-lg font-medium text-white rounded-full
                bg-gradient-to-r from-blue-600 to-blue-500"
            >
              Start Now
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
