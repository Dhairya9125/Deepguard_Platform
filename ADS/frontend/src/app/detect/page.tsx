"use client";

import React from "react";
import Navbar from "../components/Navbar";
import { Mic, Video, Image as ImageIcon, ArrowRight } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";

export default function DetectHubPage() {
  const options = [
    {
      title: "Audio Deepfake Detection",
      description: "Analyze voice recordings and speech for AI synthesis or cloning artifacts.",
      icon: <Mic className="w-8 h-8 text-cyan-400" />,
      href: "/detect/audio",
      color: "from-cyan-500/20 to-blue-600/20",
      border: "hover:border-cyan-500/50",
      glow: "group-hover:shadow-[0_0_30px_rgba(6,182,212,0.3)]",
    },
    {
      title: "Image Forgery Detection",
      description: "Detect AI-generated images, FaceSwap, and pixel-level digital manipulation.",
      icon: <ImageIcon className="w-8 h-8 text-purple-400" />,
      href: "/detect/image",
      color: "from-purple-500/20 to-pink-600/20",
      border: "hover:border-purple-500/50",
      glow: "group-hover:shadow-[0_0_30px_rgba(168,85,247,0.3)]",
    },
    {
      title: "Video Manipulation Analysis",
      description: "Scan video frames for deepfakes, lip-sync anomalies, and AI alterations.",
      icon: <Video className="w-8 h-8 text-orange-400" />,
      href: "/detect/video",
      color: "from-orange-500/20 to-red-600/20",
      border: "hover:border-orange-500/50",
      glow: "group-hover:shadow-[0_0_30px_rgba(249,115,22,0.3)]",
    },
  ];

  return (
    <div className="min-h-screen bg-[#050816] text-white font-sans overflow-hidden relative">
      {/* Background ambient glows */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-600/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-600/5 rounded-full blur-[120px]" />
      </div>

      <Navbar />

      <main className="pt-32 pb-20 px-4 md:px-6 max-w-5xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col items-center text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
            Select Analysis <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">Modality</span>
          </h1>
          <p className="text-white/50 text-lg max-w-2xl text-center w-full">
            Choose the type of media you want to scan. Our specialized engines will analyze the file and generate a comprehensive forensic report.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {options.map((option, i) => (
            <motion.div
              key={option.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * (i + 1), duration: 0.5 }}
            >
              <Link href={option.href} className="block group h-full">
                <div className={`h-full relative bg-white/[0.02] backdrop-blur-sm border border-white/10 rounded-3xl p-8 transition-all duration-300 ${option.border} ${option.glow} overflow-hidden flex flex-col`}>
                  
                  {/* Card hover gradient */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${option.color} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                  
                  <div className="relative z-10 flex-1 flex flex-col">
                    <div className="mb-6 inline-flex p-4 rounded-2xl bg-white/5 border border-white/10 group-hover:scale-110 transition-transform duration-300">
                      {option.icon}
                    </div>
                    
                    <h3 className="text-2xl font-bold text-white mb-3 tracking-tight">
                      {option.title}
                    </h3>
                    
                    <p className="text-white/50 leading-relaxed mb-8 flex-1">
                      {option.description}
                    </p>

                    <div className="flex items-center text-sm font-semibold text-white/70 group-hover:text-white transition-colors mt-auto">
                      Start Scan <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  );
}
