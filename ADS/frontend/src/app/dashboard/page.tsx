"use client";

import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import {
  Activity,
  ShieldCheck,
  AlertTriangle,
  FileText,
  ArrowRight,
  Mic,
  Video,
  ScanLine,
  Clock,
  BarChart3,
  TrendingUp,
} from "lucide-react";
import { Image as ImageIcon } from "lucide-react";
import Link from "next/link";
import { motion, AnimatePresence, useMotionValue, useTransform, animate } from "framer-motion";
import { getPlatformStats } from "../lib/api";
import type { PlatformStats, JobListItem } from "../lib/api";
import { useAuth } from "../contexts/AuthContext";

// ── Animated counter ──────────────────────────────────────────────────────────
function AnimatedCount({ target, duration = 1.4 }: { target: number; duration?: number }) {
  const motionVal = useMotionValue(0);
  const rounded   = useTransform(motionVal, (v) => Math.round(v).toLocaleString());
  const [display, setDisplay] = useState("0");

  useEffect(() => {
    const controls = animate(motionVal, target, { duration, ease: "easeOut" });
    const unsub    = rounded.on("change", (v) => setDisplay(v));
    return () => { controls.stop(); unsub(); };
  }, [target, duration, motionVal, rounded]);

  return <span>{display}</span>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function timeAgo(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60)    return `${diff}s ago`;
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function verdictDot(verdict: string | null): string {
  if (verdict === "FAKE")      return "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.7)]";
  if (verdict === "REAL")      return "bg-emerald-500 shadow-[0_0_8px_rgba(34,197,94,0.7)]";
  if (verdict === "UNCERTAIN") return "bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.7)]";
  return "bg-white/20 animate-pulse";
}

function verdictTextColor(verdict: string | null, status: string): string {
  if (verdict === "FAKE")                    return "text-red-400";
  if (verdict === "REAL")                    return "text-emerald-400";
  if (verdict === "UNCERTAIN")               return "text-yellow-400";
  if (status.toUpperCase() === "PROCESSING") return "text-blue-400";
  if (status.toUpperCase() === "FAILED")     return "text-red-400";
  return "text-white/40";
}

function verdictLabel(verdict: string | null, status: string): string {
  return verdict ?? status;
}

function modalityIconEl(modality: string) {
  const m = modality.toLowerCase();
  if (m === "image") return <ImageIcon className="w-3.5 h-3.5 text-purple-400" />;
  if (m === "audio") return <Mic className="w-3.5 h-3.5 text-cyan-400" />;
  if (m === "video") return <Video className="w-3.5 h-3.5 text-orange-400" />;
  return null;
}

// ── Skeletons ─────────────────────────────────────────────────────────────────
function StatSkeleton() {
  return (
    <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-6">
      <div className="flex justify-between items-start mb-4">
        <div className="h-4 w-24 rounded-full bg-white/5 shimmer" />
        <div className="w-10 h-10 rounded-xl bg-white/5 shimmer" />
      </div>
      <div className="h-8 w-20 rounded-full bg-white/5 shimmer mb-2" />
    </div>
  );
}

function FeedSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 p-4 rounded-xl bg-white/[0.02]">
          <div className="w-2.5 h-2.5 rounded-full bg-white/5 shimmer shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-3.5 w-28 rounded-full bg-white/5 shimmer" />
            <div className="h-2.5 w-20 rounded-full bg-white/5 shimmer" />
          </div>
          <div className="h-3 w-12 rounded-full bg-white/5 shimmer" />
        </div>
      ))}
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
interface StatCardProps {
  title: string;
  value: number;
  icon: React.ReactNode;
  accent: string;
  glowColor: string;
  delay: number;
}

function StatCard({ title, value, icon, accent, glowColor, delay }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className={"relative overflow-hidden bg-white/[0.03] border rounded-2xl p-6 backdrop-blur-xl " + accent}
    >
      {/* Glow blob */}
      <div className={"absolute -top-6 -right-6 w-24 h-24 rounded-full blur-2xl opacity-20 " + glowColor} />
      <div className="flex justify-between items-start mb-4 relative">
        <h3 className="text-white/50 text-sm font-medium">{title}</h3>
        <div className={"p-2.5 rounded-xl border " + accent}>{icon}</div>
      </div>
      <div className="text-3xl font-extrabold mb-1 relative">
        <AnimatedCount target={value} />
      </div>
    </motion.div>
  );
}

// ── Modality bar chart ────────────────────────────────────────────────────────
function ModalityChart({ data }: { data: { image: number; audio: number; video: number } }) {
  const total = (data.image + data.audio + data.video) || 1;
  const bars = [
    { label: "Image", value: data.image, color: "bg-purple-500", glow: "shadow-[0_0_8px_rgba(168,85,247,0.4)]" },
    { label: "Audio", value: data.audio, color: "bg-cyan-500",   glow: "shadow-[0_0_8px_rgba(6,182,212,0.4)]"  },
    { label: "Video", value: data.video, color: "bg-orange-500", glow: "shadow-[0_0_8px_rgba(249,115,22,0.4)]" },
  ];
  return (
    <div className="space-y-4">
      {bars.map((bar, i) => (
        <motion.div
          key={bar.label}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 + i * 0.1 }}
        >
          <div className="flex justify-between text-xs text-white/50 mb-1.5">
            <span>{bar.label}</span>
            <span className="font-medium text-white/70">{bar.value.toLocaleString()}</span>
          </div>
          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(bar.value / total) * 100}%` }}
              transition={{ delay: 0.5 + i * 0.1, duration: 0.8, ease: "easeOut" }}
              className={"h-full rounded-full " + bar.color + " " + bar.glow}
            />
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// ── Recent job feed item ──────────────────────────────────────────────────────
function FeedItem({ job, i }: { job: JobListItem; i: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.35 + i * 0.07 }}
    >
      <Link
        href={"/results/" + job.job_id}
        className="flex items-center gap-4 p-3.5 rounded-xl bg-white/[0.02] hover:bg-white/[0.05] border border-transparent hover:border-white/10 transition-all group"
      >
        <div className={"w-2.5 h-2.5 rounded-full shrink-0 " + verdictDot(job.verdict)} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-semibold text-white/70">
              {job.job_id.slice(0, 8)}
            </span>
            <span className="flex items-center gap-1 text-white/30">
              {modalityIconEl(job.modality)}
            </span>
          </div>
          <div className="text-xs text-white/30 flex items-center gap-1 mt-0.5">
            <Clock className="w-3 h-3" />
            {timeAgo(job.created_at)}
          </div>
        </div>
        <div className={"text-xs font-bold " + verdictTextColor(job.verdict, job.status)}>
          {verdictLabel(job.verdict, job.status)}
        </div>
        <ArrowRight className="w-3.5 h-3.5 text-white/20 group-hover:text-white/50 transition-colors" />
      </Link>
    </motion.div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [stats,   setStats]   = useState<PlatformStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);
  const { user } = useAuth();

  useEffect(() => {
    getPlatformStats()
      .then((data) => { setStats(data); setError(null); })
      .catch((e: Error) => setError(e.message ?? "Failed to load stats."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-[#050816] text-white font-sans overflow-x-hidden">
      {/* Ambient glows */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-60 -left-40 w-[700px] h-[700px] bg-blue-600/4 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-cyan-500/3 rounded-full blur-3xl" />
      </div>

      <Navbar />

      <main className="pt-28 pb-20 px-4 md:px-6 max-w-7xl mx-auto relative">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col md:flex-row justify-between items-start md:items-end mb-10 gap-4"
        >
          <div>
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-2">
              Welcome,{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">
                {user?.first_name || user?.username || "Admin"}
              </span>
            </h1>
            <p className="text-white/50 text-lg">Your platform overview at a glance.</p>
          </div>
          <div className="flex gap-3">
            <Link
              href="/history"
              className="px-5 py-2.5 bg-white/5 border border-white/10 hover:bg-white/10 rounded-xl font-semibold text-sm transition-colors"
            >
              View History
            </Link>
            <Link
              href="/detect"
              className="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-cyan-400 text-white font-semibold rounded-xl flex items-center gap-2 shadow-lg shadow-blue-500/25 transition-all duration-300"
            >
              <ScanLine className="w-5 h-5" />
              <span className="hidden md:inline">New Scan</span>
            </Link>
          </div>
        </motion.div>

        {/* Error banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              key="err"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6 flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-300 text-sm"
            >
              <AlertTriangle className="w-4 h-4 shrink-0" />
              {error} — stats may be unavailable.
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Stat cards ─────────────────────────────────── */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            {Array.from({ length: 4 }).map((_, i) => <StatSkeleton key={i} />)}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            <StatCard
              title="Total Scans"
              value={stats?.total_jobs ?? 0}
              icon={<FileText className="w-5 h-5 text-blue-400" />}
              accent="border-blue-500/15"
              glowColor="bg-blue-500"
              delay={0.1}
            />
            <StatCard
              title="Deepfakes Detected"
              value={stats?.total_fake ?? 0}
              icon={<AlertTriangle className="w-5 h-5 text-red-400" />}
              accent="border-red-500/15"
              glowColor="bg-red-500"
              delay={0.18}
            />
            <StatCard
              title="Authentic Media"
              value={stats?.total_real ?? 0}
              icon={<ShieldCheck className="w-5 h-5 text-emerald-400" />}
              accent="border-emerald-500/15"
              glowColor="bg-emerald-500"
              delay={0.26}
            />
            <StatCard
              title="Pending / Processing"
              value={stats?.total_pending ?? 0}
              icon={<Activity className="w-5 h-5 text-yellow-400" />}
              accent="border-yellow-500/15"
              glowColor="bg-yellow-500"
              delay={0.34}
            />
          </div>
        )}

        {/* ── Lower grid ─────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Recent jobs feed */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-2 bg-white/[0.03] border border-white/10 rounded-2xl p-6 backdrop-blur-xl"
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-bold flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                Recent Activity
              </h2>
              <Link href="/history" className="text-xs text-blue-400 hover:text-blue-300 transition-colors">
                View All →
              </Link>
            </div>

            {loading ? (
              <FeedSkeleton />
            ) : !stats?.recent_jobs?.length ? (
              <div className="flex flex-col items-center justify-center py-12 gap-3 text-white/30">
                <ScanLine className="w-8 h-8" />
                <p className="text-sm">No recent jobs yet. Start a scan!</p>
              </div>
            ) : (
              <div className="space-y-2">
                {stats.recent_jobs.slice(0, 5).map((job, i) => (
                  <FeedItem key={job.job_id} job={job} i={i} />
                ))}
              </div>
            )}
          </motion.div>

          {/* Right column */}
          <div className="space-y-6">
            {/* Modality breakdown */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.38 }}
              className="bg-white/[0.03] border border-white/10 rounded-2xl p-6 backdrop-blur-xl"
            >
              <h2 className="text-lg font-bold flex items-center gap-2 mb-5">
                <BarChart3 className="w-5 h-5 text-purple-400" />
                Modality Breakdown
              </h2>
              {loading ? (
                <div className="space-y-4">
                  {[0, 1, 2].map((i) => (
                    <div key={i} className="space-y-1.5">
                      <div className="h-3 w-20 rounded-full bg-white/5 shimmer" />
                      <div className="h-2 rounded-full bg-white/5 shimmer" />
                    </div>
                  ))}
                </div>
              ) : (
                <ModalityChart data={stats?.by_modality ?? { image: 0, audio: 0, video: 0 }} />
              )}
            </motion.div>

            {/* Quick CTA */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.44 }}
              className="relative overflow-hidden bg-gradient-to-br from-blue-900/40 to-cyan-900/20 border border-blue-500/20 rounded-2xl p-6 group hover:border-blue-500/40 transition-all"
            >
              <div className="absolute -top-8 -right-8 w-28 h-28 bg-blue-500/20 rounded-full blur-2xl pointer-events-none" />
              <div className="absolute -bottom-8 -left-8 w-20 h-20 bg-cyan-500/15 rounded-full blur-2xl pointer-events-none" />
              <p className="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-2">
                Quick Action
              </p>
              <h3 className="text-xl font-bold mb-2">Run a new scan</h3>
              <p className="text-sm text-white/50 mb-5">
                Upload audio, image, or video and get AI-powered deepfake analysis in seconds.
              </p>
              <Link
                href="/detect/audio"
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-semibold transition-colors shadow-lg shadow-blue-500/20"
              >
                <ScanLine className="w-4 h-4" /> Start Scan
              </Link>
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  );
}

