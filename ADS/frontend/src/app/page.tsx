"use client";

import { useEffect } from "react";
import dynamic from "next/dynamic";
import LoadingScreen from "./components/LoadingScreen";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import AboutAI from "./components/AboutAI";
import FeatureGrid from "./components/FeatureGrid";
import Capabilities from "./components/Capabilities";
import InteractiveShowcase from "./components/InteractiveShowcase";
import Intelligence from "./components/Intelligence";
import Statistics from "./components/Statistics";
import Testimonials from "./components/Testimonials";
import Pricing from "./components/Pricing";
import CTA from "./components/CTA";
import Footer from "./components/Footer";
import DetectionModal from "./components/DetectionModal";
import { useMousePosition } from "./hooks/useMousePosition";
import { useSmoothScroll } from "./hooks/useSmoothScroll";

const BackgroundCanvas = dynamic(
  () => import("./components/BackgroundCanvas"),
  { ssr: false }
);

export default function TruxPage() {
  useMousePosition();
  useSmoothScroll();

  useEffect(() => {
    document.body.style.background = "#050816";
    document.body.style.color = "#FFFFFF";
    return () => {
      document.body.style.background = "";
      document.body.style.color = "";
    };
  }, []);

  return (
    <main className="relative min-h-screen bg-[#050816] text-white overflow-x-hidden">
      <LoadingScreen />
      <BackgroundCanvas />
      <Navbar />
      <Hero />
      <AboutAI />
      <FeatureGrid />
      <Capabilities />
      <InteractiveShowcase />
      <Intelligence />
      <Statistics />
      <Testimonials />
      <Pricing />
      <CTA />
      <Footer />
      <DetectionModal />
    </main>
  );
}
