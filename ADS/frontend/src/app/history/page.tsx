"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import Navbar from "../components/Navbar";
import {
  Search,
  ChevronLeft,
  ChevronRight,
  History,
  ScanLine,
  RefreshCw,
  ExternalLink,
  Mic,
  Video,
  Layers,
  AlertCircle,
  Download,
} from "lucide-react";
import { Image as ImageIcon } from "lucide-react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { listJobs } from "../lib/api";
import type { JobListItem } from "../lib/api";


// ── Animation variants ────────────────────────────────────────────────────────
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
  },
};

const rowVariants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] as const } },
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatDate(iso: string): string {
  const d = new Date(iso);
  return (
    d.toLocaleDateString("en-US", { month: "short", day: "numeric" }) +
    ", " +
    d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })
  );
}

function verdictColor(verdict: string | null): string {
  if (verdict === "FAKE")      return "bg-red-500/10 border-red-500/20 text-red-400";
  if (verdict === "REAL")      return "bg-green-500/10 border-green-500/20 text-green-400";
  if (verdict === "UNCERTAIN") return "bg-yellow-500/10 border-yellow-500/20 text-yellow-400";
  return "bg-white/5 border-white/10 text-white/40";
}

function statusColor(status: string): string {
  const s = status.toUpperCase();
  if (s === "COMPLETED")  return "bg-emerald-500/10 border-emerald-500/20 text-emerald-400";
  if (s === "PROCESSING") return "bg-blue-500/10 border-blue-500/20 text-blue-400";
  if (s === "PENDING")    return "bg-yellow-500/10 border-yellow-500/20 text-yellow-400";
  if (s === "FAILED")     return "bg-red-500/10 border-red-500/20 text-red-400";
  return "bg-white/5 border-white/10 text-white/40";
}

function modalityIcon(modality: string) {
  const m = modality.toLowerCase();
  if (m === "image") return <ImageIcon className="w-3.5 h-3.5" />;
  if (m === "audio") return <Mic className="w-3.5 h-3.5" />;
  if (m === "video") return <Video className="w-3.5 h-3.5" />;
  return <Layers className="w-3.5 h-3.5" />;
}

function modalityBadgeColor(modality: string): string {
  const m = modality.toLowerCase();
  if (m === "image") return "bg-purple-500/10 border-purple-500/20 text-purple-300";
  if (m === "audio") return "bg-cyan-500/10 border-cyan-500/20 text-cyan-300";
  if (m === "video") return "bg-orange-500/10 border-orange-500/20 text-orange-300";
  return "bg-white/5 border-white/10 text-white/50";
}

function jobIdColor(status: string): string {
  const s = status.toUpperCase();
  if (s === "COMPLETED")  return "text-emerald-400";
  if (s === "PROCESSING") return "text-blue-400";
  if (s === "PENDING")    return "text-yellow-400";
  if (s === "FAILED")     return "text-red-400";
  return "text-blue-300";
}

function statusDot(status: string): string {
  const s = status.toUpperCase();
  if (s === "PROCESSING") return "w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse";
  if (s === "COMPLETED")  return "w-1.5 h-1.5 rounded-full bg-emerald-400";
  if (s === "FAILED")     return "w-1.5 h-1.5 rounded-full bg-red-400";
  return "w-1.5 h-1.5 rounded-full bg-yellow-400";
}

// ── Skeleton ──────────────────────────────────────────────────────────────────
function SkeletonRow({ i }: { i: number }) {
  return (
    <motion.tr
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: i * 0.05 }}
      className="border-b border-white/5"
    >
      {[80, 160, 70, 100, 90, 75, 60, 90].map((w, j) => (
        <td key={j} className="py-4 pl-4">
          <div className="h-3.5 rounded-full bg-white/5 shimmer" style={{ width: w }} />
        </td>
      ))}
    </motion.tr>
  );
}

// ── Filter pill ───────────────────────────────────────────────────────────────
function FilterPill({
  label,
  icon,
  active,
  onClick,
}: {
  label: string;
  icon: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-sm font-medium border transition-all duration-200 " +
        (active
          ? "bg-blue-600/20 border-blue-500/40 text-blue-300 shadow-[0_0_12px_rgba(77,124,254,0.15)]"
          : "bg-white/5 border-white/10 text-white/50 hover:bg-white/10 hover:text-white/80")
      }
    >
      {icon}
      {label}
    </button>
  );
}

const FILTERS: { label: string; value: string; icon: React.ReactNode }[] = [
  { label: "All",   value: "",      icon: <Layers className="w-3.5 h-3.5" /> },
  { label: "Image", value: "image", icon: <ImageIcon className="w-3.5 h-3.5" /> },
  { label: "Audio", value: "audio", icon: <Mic className="w-3.5 h-3.5" /> },
  { label: "Video", value: "video", icon: <Video className="w-3.5 h-3.5" /> },
];

// ── Page ──────────────────────────────────────────────────────────────────────
export default function HistoryPage() {
  const [jobs,     setJobs]     = useState<JobListItem[]>([]);
  const [total,    setTotal]    = useState(0);
  const [pages,    setPages]    = useState(1);
  const [page,     setPage]     = useState(1);
  const [modality, setModality] = useState("");
  const [search,   setSearch]   = useState("");
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchJobs = useCallback(async (pg: number, mod: string) => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    setLoading(true);
    setError(null);
    try {
      const data = await listJobs(pg, 20, mod || undefined);
      setJobs(data.items);
      setTotal(data.total);
      setPages(data.pages);
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message ?? "Unexpected error.");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // eslint-disable-next-line
  useEffect(() => { fetchJobs(page, modality); }, [page, modality, fetchJobs]);

  const displayedJobs = jobs.filter((j) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return j.job_id.toLowerCase().includes(q) || (j.file_name ?? "").toLowerCase().includes(q);
  });

  function handleFilter(val: string) { setModality(val); setPage(1); }

  function exportCSV() {
    if (jobs.length === 0) return;
    const headers = ["Job ID", "File Name", "Modality", "Date", "Status", "Verdict", "Fake Probability"];
    const rows = jobs.map(j => [
      j.job_id,
      `"${j.file_name || "unnamed"}"`,
      j.modality,
      new Date(j.created_at).toISOString(),
      j.status,
      j.verdict || "N/A",
      j.fake_probability !== null ? (j.fake_probability * 100).toFixed(1) + "%" : "N/A"
    ]);
    const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `deepguard_history_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  return (
    <div className="min-h-screen bg-[#050816] text-white font-sans overflow-x-hidden">
      {/* Ambient glows */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-[600px] h-[600px] bg-blue-600/5 rounded-full blur-3xl" />
        <div className="absolute top-1/2 right-0 w-[400px] h-[400px] bg-cyan-500/4 rounded-full blur-3xl" />
      </div>

      <Navbar />

      <main className="pt-28 pb-20 px-4 md:px-6 max-w-7xl mx-auto relative">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 gap-4"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <History className="w-5 h-5 text-blue-400" />
              </div>
              <h1 className="text-4xl font-extrabold tracking-tight">Analysis History</h1>
            </div>
            <p className="text-white/50 pl-1">
              {loading ? "Loading records…" : `${total} total job${total !== 1 ? "s" : ""} found`}
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={exportCSV}
              className="flex items-center gap-2 px-5 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl font-semibold text-sm transition-all duration-200"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
            <Link
              href="/detect/audio"
              className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 rounded-xl font-semibold text-sm transition-all duration-200 shadow-lg shadow-blue-500/20"
            >
              <ScanLine className="w-4 h-4" />
              New Scan
            </Link>
          </div>
        </motion.div>

        {/* Search + Filters */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 backdrop-blur-xl mb-4"
        >
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
              <input
                type="text"
                placeholder="Search by Job ID or filename…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 pl-11 pr-4 text-sm outline-none focus:border-blue-500/50 transition-all placeholder:text-white/25"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              {FILTERS.map((f) => (
                <FilterPill
                  key={f.value}
                  label={f.label}
                  icon={f.icon}
                  active={modality === f.value}
                  onClick={() => handleFilter(f.value)}
                />
              ))}
            </div>
          </div>
        </motion.div>

        {/* Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white/[0.03] border border-white/10 rounded-2xl backdrop-blur-xl overflow-hidden"
        >
          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                key="err"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="flex items-center gap-3 p-4 bg-red-500/10 border-b border-red-500/20"
              >
                <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                <span className="text-sm text-red-300 flex-1">{error}</span>
                <button
                  onClick={() => fetchJobs(page, modality)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 border border-red-500/30 text-red-300 text-xs font-medium hover:bg-red-500/30 transition-colors"
                >
                  <RefreshCw className="w-3 h-3" /> Retry
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="border-b border-white/[0.08] text-white/35 text-xs uppercase tracking-widest">
                  <th className="py-3.5 pl-6 pr-4 font-medium">Job ID</th>
                  <th className="py-3.5 pr-4 font-medium">File Name</th>
                  <th className="py-3.5 pr-4 font-medium">Modality</th>
                  <th className="py-3.5 pr-4 font-medium">Date</th>
                  <th className="py-3.5 pr-4 font-medium">Status</th>
                  <th className="py-3.5 pr-4 font-medium">Verdict</th>
                  <th className="py-3.5 pr-4 font-medium">Score</th>
                  <th className="py-3.5 pr-6 font-medium text-right">Action</th>
                </tr>
              </thead>

              <AnimatePresence mode="wait">
                {loading ? (
                  <motion.tbody key="skel" initial={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    {Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} i={i} />)}
                  </motion.tbody>
                ) : displayedJobs.length === 0 ? (
                  <motion.tbody key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <tr>
                      <td colSpan={8}>
                        <div className="flex flex-col items-center justify-center py-20 gap-4">
                          <div className="p-5 rounded-2xl bg-white/5 border border-white/10">
                            <ScanLine className="w-10 h-10 text-white/20" />
                          </div>
                          <div className="text-center">
                            <p className="text-lg font-semibold text-white/50 mb-1">
                              {search || modality ? "No matching results" : "No analyses yet"}
                            </p>
                            <p className="text-sm text-white/30">
                              {search || modality
                                ? "Try adjusting your search or filter."
                                : "No analyses yet. Run your first scan!"}
                            </p>
                          </div>
                          {!search && !modality && (
                            <Link
                              href="/detect/audio"
                              className="mt-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 rounded-xl font-semibold text-sm transition-colors"
                            >
                              Start First Scan →
                            </Link>
                          )}
                        </div>
                      </td>
                    </tr>
                  </motion.tbody>
                ) : (
                  <motion.tbody
                    key={"rows-" + page + "-" + modality}
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {displayedJobs.map((job) => (
                      <motion.tr
                        key={job.job_id}
                        variants={rowVariants}
                        className="border-b border-white/5 hover:bg-white/[0.03] transition-colors group"
                      >
                        <td className="py-4 pl-6 pr-4">
                          <span className={"font-mono text-xs font-semibold tracking-wider " + jobIdColor(job.status)}>
                            {job.job_id.slice(0, 8)}
                          </span>
                        </td>
                        <td className="py-4 pr-4 max-w-[200px]">
                          <span
                            className="text-white/75 text-xs truncate block"
                            title={job.file_name ?? ""}
                          >
                            {job.file_name || <em className="text-white/25 not-italic">unnamed</em>}
                          </span>
                        </td>
                        <td className="py-4 pr-4">
                          <span
                            className={
                              "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border " +
                              modalityBadgeColor(job.modality)
                            }
                          >
                            {modalityIcon(job.modality)}
                            {job.modality.charAt(0).toUpperCase() + job.modality.slice(1)}
                          </span>
                        </td>
                        <td className="py-4 pr-4 text-white/40 text-xs whitespace-nowrap">
                          {formatDate(job.created_at)}
                        </td>
                        <td className="py-4 pr-4">
                          <span
                            className={
                              "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium border " +
                              statusColor(job.status)
                            }
                          >
                            <span className={statusDot(job.status)} />
                            {job.status}
                          </span>
                        </td>
                        <td className="py-4 pr-4">
                          {job.verdict ? (
                            <span
                              className={
                                "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border " +
                                verdictColor(job.verdict)
                              }
                            >
                              {job.verdict}
                            </span>
                          ) : (
                            <span className="text-white/20 text-xs">—</span>
                          )}
                        </td>
                        <td className="py-4 pr-4">
                          {job.fake_probability !== null && job.fake_probability !== undefined ? (
                            <span
                              className={
                                "text-sm font-semibold " +
                                (job.fake_probability > 0.7
                                  ? "text-red-400"
                                  : job.fake_probability > 0.4
                                  ? "text-yellow-400"
                                  : "text-emerald-400")
                              }
                            >
                              {(job.fake_probability * 100).toFixed(1)}%
                            </span>
                          ) : (
                            <span className="text-white/20 text-xs">—</span>
                          )}
                        </td>
                        <td className="py-4 pr-6 text-right">
                          <Link
                            href={"/results/" + job.job_id}
                            className="inline-flex items-center gap-1 text-xs font-medium text-white/30 group-hover:text-blue-400 transition-colors"
                          >
                            View Report <ExternalLink className="w-3 h-3" />
                          </Link>
                        </td>
                      </motion.tr>
                    ))}
                  </motion.tbody>
                )}
              </AnimatePresence>
            </table>
          </div>

          {/* Pagination */}
          {!loading && displayedJobs.length > 0 && (
            <div className="flex items-center justify-between px-6 py-4 border-t border-white/[0.08] text-sm text-white/40">
              <span>
                Page {page} of {pages}&nbsp;&middot;&nbsp;{total} total
              </span>
              <div className="flex items-center gap-2">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="p-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                {Array.from({ length: Math.min(5, pages) }, (_, i) => {
                  const pg = pages <= 5 ? i + 1 : Math.max(1, Math.min(page - 2, pages - 4)) + i;
                  return (
                    <button
                      key={pg}
                      onClick={() => setPage(pg)}
                      className={
                        "w-8 h-8 rounded-lg text-xs font-medium transition-all " +
                        (pg === page
                          ? "bg-blue-600/30 border border-blue-500/40 text-blue-300"
                          : "bg-white/5 border border-white/10 text-white/40 hover:bg-white/10 hover:text-white")
                      }
                    >
                      {pg}
                    </button>
                  );
                })}
                <button
                  disabled={page >= pages}
                  onClick={() => setPage((p) => Math.min(pages, p + 1))}
                  className="p-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </motion.div>
      </main>
    </div>
  );
}

