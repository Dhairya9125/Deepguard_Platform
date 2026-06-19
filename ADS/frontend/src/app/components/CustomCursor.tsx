"use client";

import React, { useEffect, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

export default function CustomCursor() {
  const [variant, setVariant] = useState("default");
  
  const mouseX = useMotionValue(-100);
  const mouseY = useMotionValue(-100);

  // Smooth springs for the cursor follow effect
  const springConfig = { damping: 25, stiffness: 400, mass: 0.5 };
  const cursorX = useSpring(mouseX, springConfig);
  const cursorY = useSpring(mouseY, springConfig);

  useEffect(() => {
    document.body.style.cursor = "none";
    const style = document.createElement("style");
    style.innerHTML = `* { cursor: none !important; }`;
    document.head.appendChild(style);

    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);

      const target = e.target as HTMLElement;
      const closestLink = target.closest("a, button, [role='button'], input, textarea");
      const closestDetected = target.closest(".result-detected");
      const closestAuthentic = target.closest(".result-authentic");

      if (closestDetected) {
        setVariant("detected");
      } else if (closestAuthentic) {
        setVariant("authentic");
      } else if (closestLink) {
        setVariant("pointer");
      } else {
        setVariant("default");
      }
    };

    window.addEventListener("mousemove", handleMouseMove);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      document.body.style.cursor = "auto";
      document.head.removeChild(style);
    };
  }, [mouseX, mouseY]);

  // Framer motion variants for the outer SVG
  const variants = {
    default: {
      scale: 1,
      rotate: 0,
      opacity: 0.8,
      filter: "drop-shadow(0px 0px 8px rgba(102, 227, 255, 0.4))",
    },
    pointer: {
      scale: 1.5,
      rotate: 45,
      opacity: 1,
      filter: "drop-shadow(0px 0px 12px rgba(102, 227, 255, 0.8))",
    },
    detected: {
      scale: 1.6,
      rotate: 45,
      opacity: 1,
      filter: "drop-shadow(0px 0px 15px rgba(239, 68, 68, 0.8))",
    },
    authentic: {
      scale: 1.6,
      rotate: -45,
      opacity: 1,
      filter: "drop-shadow(0px 0px 15px rgba(34, 197, 94, 0.8))",
    }
  };

  // Determine colors based on variant
  const getColors = () => {
    switch (variant) {
      case "detected": return { primary: "#ef4444", secondary: "rgba(239,68,68,0.3)" };
      case "authentic": return { primary: "#22c55e", secondary: "rgba(34,197,94,0.3)" };
      case "pointer": return { primary: "#ffffff", secondary: "rgba(102,227,255,0.4)" };
      default: return { primary: "#66e3ff", secondary: "rgba(102,227,255,0.1)" };
    }
  };

  const colors = getColors();

  return (
    <>
      {/* Outer animated geometric triangle */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[99999] flex items-center justify-center mix-blend-screen"
        style={{
          x: cursorX,
          y: cursorY,
          translateX: "-50%",
          translateY: "-50%",
          width: 60,
          height: 60,
        }}
        variants={variants}
        animate={variant}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
      >
        <svg width="60" height="60" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Main Diamond */}
          <motion.polygon 
            points="50,10 90,50 50,90 10,50" 
            stroke={colors.primary} 
            strokeWidth="2" 
            fill={colors.secondary}
            animate={{ stroke: colors.primary, fill: colors.secondary }}
            transition={{ duration: 0.3 }}
          />
          {/* Inner Dashed Tracking Ring (Diamond) */}
          <motion.polygon 
            points="50,25 75,50 50,75 25,50" 
            stroke={colors.primary} 
            strokeWidth="1.5" 
            strokeDasharray="4 4"
            fill="none"
            animate={{ stroke: colors.primary }}
            transition={{ duration: 0.3 }}
          />
          {/* Sci-fi Crosshairs */}
          <motion.line x1="50" y1="0" x2="50" y2="10" stroke={colors.primary} strokeWidth="1.5" animate={{ stroke: colors.primary }} transition={{ duration: 0.3 }} />
          <motion.line x1="50" y1="90" x2="50" y2="100" stroke={colors.primary} strokeWidth="1.5" animate={{ stroke: colors.primary }} transition={{ duration: 0.3 }} />
          <motion.line x1="0" y1="50" x2="10" y2="50" stroke={colors.primary} strokeWidth="1.5" animate={{ stroke: colors.primary }} transition={{ duration: 0.3 }} />
          <motion.line x1="90" y1="50" x2="100" y2="50" stroke={colors.primary} strokeWidth="1.5" animate={{ stroke: colors.primary }} transition={{ duration: 0.3 }} />

          {/* Corner Node Accents */}
          <motion.circle cx="50" cy="10" r="3" fill={colors.primary} animate={{ fill: colors.primary }} transition={{ duration: 0.3 }} />
          <motion.circle cx="90" cy="50" r="3" fill={colors.primary} animate={{ fill: colors.primary }} transition={{ duration: 0.3 }} />
          <motion.circle cx="50" cy="90" r="3" fill={colors.primary} animate={{ fill: colors.primary }} transition={{ duration: 0.3 }} />
          <motion.circle cx="10" cy="50" r="3" fill={colors.primary} animate={{ fill: colors.primary }} transition={{ duration: 0.3 }} />
        </svg>
      </motion.div>

      {/* Inner precise dot with double-layered glow */}
      <motion.div
        className="fixed top-0 left-0 flex items-center justify-center pointer-events-none z-[100000]"
        style={{
          x: mouseX,
          y: mouseY,
          translateX: "-50%",
          translateY: "-50%",
        }}
      >
        <motion.div
          className="absolute w-4 h-4 rounded-full"
          style={{ backgroundColor: colors.secondary, filter: "blur(2px)" }}
        />
        <motion.div
          className="absolute w-1.5 h-1.5 rounded-full"
          style={{ backgroundColor: colors.primary }}
        />
      </motion.div>
    </>
  );
}
