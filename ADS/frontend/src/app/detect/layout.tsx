"use client";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import DetectionModal from "../components/DetectionModal";

export default function DetectLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <main className="relative min-h-screen bg-[#050816] text-white overflow-x-hidden">
      <Navbar />
      {children}
      <Footer />
      <DetectionModal />
    </main>
  );
}
