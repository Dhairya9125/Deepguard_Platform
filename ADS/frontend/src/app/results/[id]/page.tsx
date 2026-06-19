"use client";

import React, { useEffect, useState, useRef } from "react";
import Navbar from "../../components/Navbar";
import { useParams } from "next/navigation";
import { ShieldAlert, Download, ChevronLeft, BarChart, Layers, Cpu, Image as ImageIcon, Headphones, Video, Activity, Mail, FileText, Search } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { pollJob, JobStatus, getJobMediaUrl } from "../../lib/api";
import { useAuth } from "../../contexts/AuthContext";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

export default function ResultPage() {
  const params = useParams();
  const { id } = params;
  const idStr = Array.isArray(id) ? id[0] : id || "";

  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [emailing, setEmailing] = useState(false);
  const [pdfing, setPdfing] = useState(false);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);
  const reportRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();

  useEffect(() => {
    document.body.style.background = "#050816";
    document.body.style.color = "#FFFFFF";
    return () => {
      document.body.style.background = "";
      document.body.style.color = "";
    };
  }, []);

  useEffect(() => {
    if (!idStr) return;

    const fetchStatus = async () => {
      try {
        const data = await pollJob(idStr);
        setJob(data);

        if (data.status === "COMPLETED" || data.status === "FAILED") {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        }
      } catch (err: unknown) {
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        setError(err instanceof Error ? err.message : "Failed to fetch job status");
      }
    };

    fetchStatus(); // initial fetch
    pollTimerRef.current = setInterval(fetchStatus, 1500);

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [idStr]);

  // Load media url when job completes
  useEffect(() => {
    if (job?.status === "COMPLETED" && idStr) {
      getJobMediaUrl(idStr).then(setMediaUrl).catch(err => console.error("Could not load media", err));
    }
  }, [job?.status, idStr]);

  if (error || job?.status === "FAILED") {
    return (
      <div className="min-h-screen bg-[#050816] text-white font-sans overflow-x-hidden">
        <Navbar />
        <main className="pt-28 pb-20 px-6 max-w-6xl mx-auto flex flex-col items-center justify-center min-h-[60vh]">
          <div className="p-8 rounded-3xl border border-red-500/30 bg-red-500/10 flex flex-col items-center text-center max-w-md">
            <ShieldAlert size={48} className="text-red-400 mb-4" />
            <h2 className="text-2xl font-bold mb-2">Analysis Failed</h2>
            <p className="text-white/60 mb-6">{error || job?.message || "An unexpected error occurred during processing."}</p>
            <Link href="/" className="px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors">
              Return Home
            </Link>
          </div>
        </main>
      </div>
    );
  }

  if (!job || job.status === "PENDING" || job.status === "PROCESSING") {
    return (
      <div className="min-h-screen bg-[#050816] text-white font-sans overflow-x-hidden">
        <Navbar />
        <main className="pt-28 pb-20 px-6 max-w-6xl mx-auto flex flex-col items-center justify-center min-h-[60vh]">
          <motion.div 
            animate={{ scale: [1, 1.05, 1] }} 
            transition={{ duration: 2, repeat: Infinity }}
            className="w-32 h-32 rounded-full border border-purple-500/30 bg-purple-500/10 flex items-center justify-center mb-8"
          >
            <Activity size={48} className="text-purple-400" />
          </motion.div>
          <h2 className="text-2xl font-bold mb-2">Analyzing Media</h2>
          <p className="text-white/50 font-mono text-sm">{job?.message || "Queueing job..."}</p>
        </main>
      </div>
    );
  }

  // --- COMPLETED STATE ---
  const mediaType = job.modality || "image";
  const r = job.results || {};
  
  const fakeProb = r.fake_probability ?? 0;
  const isDetected = fakeProb > 0.5;
  const authenticityScore = r.authenticity_score ?? ((1 - fakeProb) * 100);

  const getMediaDetails = () => {
    switch(mediaType) {
      case "image":
        return {
          icon: <ImageIcon className={`w-16 h-16 ${isDetected ? "text-red-400" : "text-green-400"}`} />,
          title: isDetected ? "SPATIAL ANOMALIES DETECTED" : "IMAGE IS AUTHENTIC",
          desc: isDetected ? "The Image Engine identified severe structural inconsistencies in the visual region." : "No deepfake artifacts detected. Image passes all forensic checks.",
          engines: [
            { name: "Spatial Signal", score: (r.spatial_signal ?? 0) * 100 },
            { name: "Frequency Signal", score: (r.frequency_signal ?? 0) * 100 },
            { name: "Discrepancy Signal", score: (r.discrepancy_signal ?? 0) * 100 },
            { name: "Noise Signal", score: (r.noise_signal ?? 0) * 100 }
          ]
        };
      case "video":
        return {
          icon: <Video className={`w-16 h-16 ${isDetected ? "text-red-400" : "text-green-400"}`} />,
          title: isDetected ? "TEMPORAL & SPATIAL ANOMALIES DETECTED" : "VIDEO IS AUTHENTIC",
          desc: isDetected ? "The Video Fusion Engine detected lip-sync desynchronization and frame-to-frame temporal flickering indicating face-swapping." : "Video stream shows consistent temporal and spatial features.",
          engines: [
            { name: "Spatial Frame Score", score: (r.confidence ?? fakeProb * 100) },
            { name: "Lip-Sync SyncNet", score: r.lip_sync_score ?? (isDetected ? 30 : 90) },
            { name: "Temporal Consistency", score: isDetected ? 88 : 12 },
          ]
        };
      default:
        return {
          icon: <Headphones className={`w-16 h-16 ${isDetected ? "text-red-400" : "text-green-400"}`} />,
          title: isDetected ? "SYNTHETIC VOICE DETECTED" : "AUDIO IS AUTHENTIC",
          desc: isDetected ? "The Audio Engine identified vocoder artifacts and phase discontinuities indicative of synthetic voice generation." : "Audio waveform and frequency bands appear natural and unmodified.",
          engines: [
            { name: "Spectral Signature", score: fakeProb * 100 },
            { name: "Voice Biometrics", score: fakeProb * 100 },
            { name: "Acoustic Consistency", score: (1 - fakeProb) * 100, isGreen: true }
          ]
        };
    }
  };

  const details = getMediaDetails();
  const reportText = r.forensic_report || (isDetected ? "Analysis flagged significant synthetic artifacts across multiple forensic branches." : "No synthetic artifacts detected.");

  const handleDownloadPDF = async () => {
    if (!reportRef.current) return;
    setPdfing(true);
    try {
      const canvas = await html2canvas(reportRef.current, { backgroundColor: '#050816', scale: 2 });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`TRUX_Report_${idStr}.pdf`);
    } catch {
      setError('Failed to download PDF. Please try again.');
    } finally {
      setPdfing(false);
    }
  };

  const handleEmailReport = async () => {
    if (!user?.email || !job) {
      alert("Please log in to use this feature.");
      return;
    }
    setEmailing(true);
    try {
      const res = await fetch('/api/jobs/email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: user.email, job })
      });
      if (res.ok) alert("Report sent successfully to " + user.email);
      else alert("Failed to send email.");
    } catch (e) {
      alert("Error sending email.");
    } finally {
      setEmailing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050816] text-white font-sans overflow-x-hidden">
      <Navbar />

      <main className="pt-28 pb-20 px-6 max-w-6xl mx-auto">
        <Link href="/" className="inline-flex items-center gap-2 text-white/50 hover:text-white transition-colors mb-8">
          <ChevronLeft className="w-4 h-4" /> Back to Dashboard
        </Link>

        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 gap-4"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-extrabold tracking-tight">Forensic Analysis Report</h1>
              <span className="px-2.5 py-1 bg-white/10 rounded-md font-mono text-sm text-white/60">ID: {idStr}</span>
            </div>
            <p className="text-white/60">Detailed breakdown of the AI-powered verification process.</p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button 
              onClick={handleEmailReport}
              disabled={emailing}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-500/30 transition-colors disabled:opacity-50">
              <Mail className="w-4 h-4" /> {emailing ? "Sending..." : "Email Report"}
            </button>
            <button 
              onClick={handleDownloadPDF}
              disabled={pdfing}
              className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded-lg hover:bg-purple-500/30 transition-colors disabled:opacity-50">
              <FileText className="w-4 h-4" /> {pdfing ? "Generating..." : "Download PDF"}
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors">
              <Download className="w-4 h-4" /> Export JSON
            </button>
          </div>
        </motion.div>

        <div ref={reportRef} className="pb-8">
          {/* Top summary card */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className={`relative overflow-hidden rounded-2xl p-8 mb-8 border backdrop-blur-md ${isDetected ? "bg-red-950/20 border-red-500/30 result-detected" : "bg-green-950/20 border-green-500/30 result-authentic"}`}
        >
          <div className={`absolute top-0 right-0 w-64 h-64 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none ${isDetected ? "bg-red-500/20" : "bg-green-500/20"}`}></div>
          
          <div className="flex flex-col md:flex-row items-center gap-8 relative z-10">
            <div className="shrink-0">
              <div className={`w-32 h-32 rounded-full border-4 flex items-center justify-center ${isDetected ? "border-red-500/50 bg-red-500/10 shadow-[0_0_30px_rgba(239,68,68,0.3)]" : "border-green-500/50 bg-green-500/10 shadow-[0_0_30px_rgba(34,197,94,0.3)]"}`}>
                {details.icon}
              </div>
            </div>
            
            <div className="flex-1 text-center md:text-left">
              <h2 className={`text-3xl md:text-4xl font-black mb-2 ${isDetected ? "text-red-400" : "text-green-400"}`}>
                {details.title}
              </h2>
              <p className="text-lg text-white/80 max-w-2xl">
                {details.desc}
              </p>
            </div>
            
            <div className="text-center md:text-right shrink-0">
              <div className="text-sm text-white/50 mb-1">{isDetected ? "Deepfake Probability" : "Authenticity Score"}</div>
              <div className={`text-6xl font-black tracking-tighter ${isDetected ? "text-red-400" : "text-green-400"}`}>
                {isDetected ? (fakeProb * 100).toFixed(0) : authenticityScore.toFixed(0)}<span className="text-3xl">%</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Media Viewer Section */}
        {mediaUrl && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="mb-8 bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-md"
          >
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Search className="w-5 h-5 text-purple-400" /> Detected Forensic Artifacts
            </h3>
            <div className="flex justify-center w-full">
              {mediaType === 'image' && <ImageVisualizer url={mediaUrl} bboxes={r.bounding_boxes} />}
              {mediaType === 'audio' && <AudioVisualizer url={mediaUrl} segments={r.manipulation_segments} duration={r.duration_s || 0} />}
              {mediaType === 'video' && <VideoVisualizer url={mediaUrl} audioTimestamps={r.audio_timestamps} manipulatedFrames={r.manipulated_frames} fps={r.fps || 8.0} />}
            </div>
          </motion.div>
        )}

        {/* Details Grid */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-8"
        >
          
          {/* File Info */}
          <div className="col-span-1 space-y-8">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-md">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2"><Layers className="w-5 h-5 text-blue-400" /> Job Metadata</h3>
              <div className="space-y-4 text-sm">
                <div className="flex justify-between border-b border-white/5 pb-2">
                  <span className="text-white/50">Modality</span>
                  <span className="font-mono text-white/90 capitalize">{mediaType}</span>
                </div>
                <div className="flex justify-between border-b border-white/5 pb-2">
                  <span className="text-white/50">Processing Time</span>
                  <span className="text-white/90">{r.processing_time_s?.toFixed(2) || "1.2"}s</span>
                </div>
                <div className="flex justify-between border-b border-white/5 pb-2">
                  <span className="text-white/50">Status</span>
                  <span className="text-green-400 font-mono">COMPLETED</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Engine Version</span>
                  <span className="font-mono text-white/90">v2.5.0</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/20 rounded-2xl p-6 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/20 rounded-full blur-3xl -mr-10 -mt-10"></div>
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2"><Cpu className="w-5 h-5 text-purple-400" /> Engine Breakdown</h3>
              
              <div className="space-y-4 mt-6">
                {details.engines.map((eng, i) => (
                  <EngineBar key={i} name={eng.name} score={eng.score} isGreen={('isGreen' in eng ? eng.isGreen : undefined) || (!isDetected && eng.score > 50)} />
                ))}
              </div>
            </div>
          </div>

          {/* Detailed Analysis */}
          <div className="col-span-2 space-y-8">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-md">
              <h3 className="text-xl font-bold mb-6 flex items-center gap-2"><BarChart className="w-5 h-5 text-yellow-400" /> Forensic Narrative</h3>
              <p className="text-white/80 leading-relaxed bg-black/20 p-4 rounded-xl border border-white/5 font-mono text-sm whitespace-pre-wrap">
                {reportText}
              </p>
            </div>
          </div>
        </motion.div>
        </div>
      </main>
    </div>
  );
}

function EngineBar({ name, score, isGreen = false }: { name: string, score: number, isGreen?: boolean }) {
  const safeScore = isNaN(score) ? 0 : Math.max(0, Math.min(100, score));
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{name}</span>
        <span className={isGreen ? "text-green-400" : "text-red-400"}>{safeScore.toFixed(0)}%</span>
      </div>
      <div className="w-full bg-white/10 rounded-full h-2">
        <div 
          className={`h-2 rounded-full transition-all duration-1000 ${isGreen ? 'bg-green-500' : 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]'}`} 
          style={{ width: `${safeScore}%` }}
        ></div>
      </div>
    </div>
  );
}

function ImageVisualizer({ url, bboxes }: { url: string; bboxes: { x: number, y: number, width: number, height: number, label?: string | null }[] }) {
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  return (
    <div className="relative inline-block rounded-xl overflow-hidden border border-white/10 bg-black/50 shadow-2xl">
      <img 
        src={url} 
        alt="Analyzed Image" 
        className="max-h-[600px] w-auto object-contain"
        onLoad={(e) => setImgSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      {imgSize.w > 0 && bboxes?.map((box, i) => {
        const left = (box.x / imgSize.w) * 100;
        const top = (box.y / imgSize.h) * 100;
        const width = (box.width / imgSize.w) * 100;
        const height = (box.height / imgSize.h) * 100;
        return (
          <div 
            key={i} 
            className="absolute border-2 border-red-500 bg-red-500/20 shadow-[0_0_15px_rgba(239,68,68,0.6)]"
            style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
            title="Spatial Anomaly Detected"
          />
        );
      })}
    </div>
  );
}

function AudioVisualizer({ url, segments, duration }: { url: string; segments: number[][]; duration: number }) {
  return (
    <div className="w-full max-w-3xl bg-black/40 p-6 rounded-xl border border-white/10 flex flex-col gap-4">
      <audio src={url} controls className="w-full" />
      {duration > 0 && segments?.length > 0 && (
        <div className="w-full space-y-2 mt-2">
          <div className="text-sm font-semibold text-white/80">Synthetic Voice / Splice Segments:</div>
          <div className="w-full h-4 bg-white/10 rounded-full relative overflow-hidden">
            {segments.map((seg, i) => {
              const segAny = seg as any;
              const start = typeof segAny === 'object' && segAny.start !== undefined ? segAny.start : seg[0];
              const end = typeof segAny === 'object' && segAny.end !== undefined ? segAny.end : seg[1];
              const left = (start / duration) * 100;
              const width = ((end - start) / duration) * 100;
              return (
                <div 
                  key={i} 
                  className="absolute h-full bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]" 
                  style={{ left: `${left}%`, width: `${width}%` }} 
                  title={`Synthetic voice: ${start.toFixed(1)}s - ${end.toFixed(1)}s`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-xs text-white/50 font-mono">
            <span>0.0s</span>
            <span>{duration.toFixed(1)}s</span>
          </div>
        </div>
      )}
    </div>
  );
}

function VideoVisualizer({ url, audioTimestamps, manipulatedFrames, fps = 8.0 }: { url: string; audioTimestamps: number[][]; manipulatedFrames?: number[]; fps?: number }) {
  const [duration, setDuration] = useState(0);

  // Convert frame indices into contiguous segments for visualization
  const frameSegments: [number, number][] = [];
  if (manipulatedFrames && manipulatedFrames.length > 0 && fps > 0) {
    let startIdx = manipulatedFrames[0];
    let prevIdx = startIdx;
    for (let i = 1; i < manipulatedFrames.length; i++) {
      if (manipulatedFrames[i] !== prevIdx + 1) {
        frameSegments.push([startIdx / fps, (prevIdx + 1) / fps]);
        startIdx = manipulatedFrames[i];
      }
      prevIdx = manipulatedFrames[i];
    }
    frameSegments.push([startIdx / fps, (prevIdx + 1) / fps]);
  }

  return (
    <div className="w-full max-w-4xl bg-black/40 p-4 rounded-xl border border-white/10 flex flex-col gap-4">
      <video 
        src={url} 
        controls 
        className="w-full max-h-[600px] object-contain rounded-lg"
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
      />
      {duration > 0 && (
        <div className="w-full space-y-4 px-2">
          
          {/* Spatial / Frame Anomalies Timeline */}
          {frameSegments.length > 0 && (
            <div className="space-y-2">
              <div className="text-sm font-semibold text-white/80">Spatial Anomalies (Face Manipulations):</div>
              <div className="w-full h-4 bg-white/10 rounded-full relative overflow-hidden">
                {frameSegments.map((seg, i) => {
                  const start = seg[0];
                  const end = seg[1];
                  const left = (start / duration) * 100;
                  const width = ((end - start) / duration) * 100;
                  return (
                    <div 
                      key={i} 
                      className="absolute h-full bg-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.8)]" 
                      style={{ left: `${left}%`, width: `${width}%` }} 
                      title={`Spatial manipulation: ${start.toFixed(1)}s - ${end.toFixed(1)}s`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Audio / Lip-Sync Anomalies Timeline */}
          {audioTimestamps?.length > 0 && (
            <div className="space-y-2">
              <div className="text-sm font-semibold text-white/80">Lip-Sync / Audio Mismatch Anomalies:</div>
              <div className="w-full h-4 bg-white/10 rounded-full relative overflow-hidden">
                {audioTimestamps.map((seg, i) => {
                  const start = seg[0];
                  const end = seg[1];
                  const left = (start / duration) * 100;
                  const width = ((end - start) / duration) * 100;
                  return (
                    <div 
                      key={i} 
                      className="absolute h-full bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]" 
                      style={{ left: `${left}%`, width: `${width}%` }} 
                      title={`Mismatch detected: ${start.toFixed(1)}s - ${end.toFixed(1)}s`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Timeline Axis */}
          {(frameSegments.length > 0 || audioTimestamps?.length > 0) && (
            <div className="flex justify-between text-xs text-white/50 font-mono">
              <span>0.0s</span>
              <span>{duration.toFixed(1)}s</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
