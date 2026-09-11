import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TRUX — AI-Powered Deepfake Detection Platform",
  description:
    "Enterprise-grade deepfake detection for image, audio, and video forensics. Protect truth with advanced AI.",
};

export default function TruxLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="antialiased">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
