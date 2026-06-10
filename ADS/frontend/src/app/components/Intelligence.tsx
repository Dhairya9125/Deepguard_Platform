"use client";

import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

const tabs = ["Analytics", "Model Monitoring"] as const;
type Tab = (typeof tabs)[number];

const chartColors = {
  blue: "#4D7CFE",
  cyan: "#66E3FF",
  purple: "#A855F7",
  green: "#34D399",
  orange: "#FB923C",
  red: "#F87171",
  yellow: "#FBBF24",
};

function LineTrendChart({ inView }: { inView: boolean }) {
  const data = [65, 72, 68, 85, 78, 92, 88];
  const labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const w = 360;
  const h = 150;
  const max = Math.max(...data);
  const points = data.map((v, i) => ({
    x: (i / (data.length - 1)) * w,
    y: h - (v / max) * h * 0.85 - 5,
  }));
  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h + 20}`} className="w-full h-full">
      <motion.path
        d={pathD}
        fill="none"
        stroke={chartColors.blue}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={inView ? { pathLength: 1 } : {}}
        transition={{ duration: 1.5, delay: 0.3, ease: "easeInOut" }}
      />
      <motion.path
        d={`${pathD} L ${points[points.length - 1].x} ${h + 20} L ${points[0].x} ${h + 20} Z`}
        fill="url(#trendGrad)"
        opacity="0.3"
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 0.3 } : {}}
        transition={{ duration: 1, delay: 0.5 }}
      />
      {points.map((p, i) => (
        <motion.circle
          key={i}
          cx={p.x}
          cy={p.y}
          r="4"
          fill={chartColors.blue}
          initial={{ opacity: 0, scale: 0 }}
          animate={inView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.4, delay: 0.6 + i * 0.08 }}
        />
      ))}
      {points.map((p, i) => (
        <text key={i} x={p.x} y={h + 15} textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="10">
          {labels[i]}
        </text>
      ))}
      <defs>
        <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={chartColors.blue} stopOpacity="0.4" />
          <stop offset="100%" stopColor={chartColors.blue} stopOpacity="0" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function BarChart({ inView }: { inView: boolean }) {
  const data = [
    { label: "Face", value: 92 },
    { label: "Audio", value: 78 },
    { label: "Video", value: 65 },
    { label: "Text", value: 44 },
    { label: "Hybrid", value: 31 },
  ];
  const max = Math.max(...data.map((d) => d.value));

  return (
    <div className="flex items-end justify-between gap-3 h-36">
      {data.map((d, i) => (
        <div key={d.label} className="flex flex-col items-center gap-1 flex-1">
          <motion.div
            className="w-full rounded-t-md"
            style={{
              background: `linear-gradient(to top, ${chartColors.blue}, ${chartColors.cyan})`,
            }}
            initial={{ height: 0 }}
            animate={inView ? { height: `${(d.value / max) * 100}%` } : {}}
            transition={{ duration: 0.8, delay: 0.3 + i * 0.1, ease: "easeOut" }}
          >
            <motion.div
              className="w-full bg-white/10"
              initial={{ height: "0%" }}
              animate={inView ? { height: "30%" } : {}}
              transition={{ duration: 0.4, delay: 0.4 + i * 0.1 }}
            />
          </motion.div>
          <span className="text-[10px] text-white/40">{d.value}%</span>
          <span className="text-[8px] text-white/30">{d.label}</span>
        </div>
      ))}
    </div>
  );
}

function DonutChart({ inView }: { inView: boolean }) {
  const segments = [
    { label: "StyleGAN", value: 34, color: chartColors.blue },
    { label: "Diffusion", value: 28, color: chartColors.cyan },
    { label: "VAE", value: 18, color: chartColors.purple },
    { label: "Adversarial", value: 12, color: chartColors.orange },
    { label: "Other", value: 8, color: chartColors.green },
  ];
  const total = segments.reduce((s, seg) => s + seg.value, 0);
  let cumulative = 0;
  const cx = 100;
  const cy = 100;
  const r = 44;
  const ir = 30;

  const polarToCart = (angle: number) => {
    const a = ((angle - 90) * Math.PI) / 180;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  };

  const arcPath = (startAngle: number, endAngle: number) => {
    const start = polarToCart(startAngle);
    const end = polarToCart(endAngle);
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  };

  const innerArc = (startAngle: number, endAngle: number) => {
    const is = polarToCart(startAngle);
    const ie = polarToCart(endAngle);
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    const ir2 = ir;
    return `M ${is.x} ${is.y} A ${ir2} ${ir2} 0 ${largeArc} 0 ${ie.x} ${ie.y}`;
  };

  return (
    <div className="flex items-center gap-6">
      <svg viewBox="0 0 200 200" className="w-44 h-44 flex-shrink-0">
        {segments.map((seg) => {
          const startAngle = (cumulative / total) * 360;
          cumulative += seg.value;
          const endAngle = (cumulative / total) * 360;
          const outer = arcPath(startAngle, endAngle);
          const inner = innerArc(startAngle, endAngle);
          const slicePath = `${outer} L ${polarToCart(endAngle).x} ${polarToCart(endAngle).y} L ${polarToCart(startAngle).x} ${polarToCart(startAngle).y} Z`;

          return (
            <motion.path
              key={seg.label}
              d={slicePath}
              fill={seg.color}
              opacity="0.85"
              initial={{ opacity: 0 }}
              animate={inView ? { opacity: 0.85 } : {}}
              transition={{ duration: 0.5 }}
            />
          );
        })}
        <circle cx={cx} cy={cy} r={ir} fill="#050816" />
        <motion.text
          x={cx}
          y={cx - 4}
          textAnchor="middle"
          fill="white"
          fontSize="18"
          fontWeight="bold"
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          {total}
        </motion.text>
        <motion.text
          x={cx}
          y={cx + 14}
          textAnchor="middle"
          fill="rgba(255,255,255,0.4)"
          fontSize="9"
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          types
        </motion.text>
      </svg>
      <div className="space-y-2">
        {segments.map((seg) => (
          <div key={seg.label} className="flex items-center gap-2.5 text-sm">
            <span className="w-3 h-3 rounded-full" style={{ background: seg.color }} />
            <span className="text-white/60">{seg.label}</span>
            <span className="text-white/40 font-mono ml-auto font-semibold">{seg.value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AreaChart({ inView }: { inView: boolean }) {
  const data = [45, 52, 48, 62, 58, 70, 67, 73, 68, 75, 72, 78];
  const w = 360;
  const h = 130;
  const max = Math.max(...data);
  const points = data.map((v, i) => ({
    x: (i / (data.length - 1)) * w,
    y: h - (v / max) * h * 0.8 - 5,
  }));
  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h + 10}`} className="w-full h-full">
      <motion.path
        d={`${pathD} L ${points[points.length - 1].x} ${h + 10} L ${points[0].x} ${h + 10} Z`}
        fill="url(#areaGrad)"
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 0.4 } : {}}
        transition={{ duration: 1, delay: 0.3 }}
      />
      <motion.path
        d={pathD}
        fill="none"
        stroke={chartColors.cyan}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={inView ? { pathLength: 1 } : {}}
        transition={{ duration: 1.5, delay: 0.4, ease: "easeInOut" }}
      />
      {points.filter((_, i) => i % 2 === 0).map((p, i) => (
        <motion.circle
          key={i}
          cx={p.x}
          cy={p.y}
          r="3"
          fill={chartColors.cyan}
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.3, delay: 0.8 + i * 0.1 }}
        />
      ))}
      <defs>
        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={chartColors.cyan} stopOpacity="0.5" />
          <stop offset="100%" stopColor={chartColors.cyan} stopOpacity="0" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function MiniLatencyChart({ inView }: { inView: boolean }) {
  const data = [120, 135, 118, 142, 128, 145, 132, 138, 125, 140, 130, 136];
  const w = 360;
  const h = 90;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full">
      {data.map((v, i) => {
        const x = (i / (data.length - 1)) * w;
        const barH = ((v - min) / range) * (h - 8);
        return (
          <motion.rect
            key={i}
            x={x - 2.5}
            y={h - barH - 4}
            width="5"
            height={barH}
            rx="2"
            fill={chartColors.green}
            initial={{ height: 0, y: h }}
            animate={inView ? { height: barH, y: h - barH - 4 } : {}}
            transition={{ duration: 0.5, delay: 0.2 + i * 0.05, ease: "easeOut" }}
          />
        );
      })}
      <motion.line
        x1="0"
        y1={h / 2}
        x2={w}
        y2={h / 2}
        stroke="rgba(255,255,255,0.1)"
        strokeWidth="0.5"
        strokeDasharray="3 3"
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 1 } : {}}
        transition={{ duration: 0.5, delay: 0.8 }}
      />
    </svg>
  );
}

const analyticsCards = [
  { label: "Total Detections (30d)", value: "847,293", color: chartColors.blue },
  { label: "Avg Accuracy", value: "98.7%", color: chartColors.green },
  { label: "False Positive Rate", value: "0.03%", color: chartColors.orange },
  { label: "OOD Events", value: "12", color: chartColors.purple },
];

const monitorCards = [
  { label: "Inference Latency", value: "142ms", color: chartColors.cyan },
  { label: "GPU Utilization", value: "67%", color: chartColors.blue },
  { label: "Model Drift", value: "0.008", color: chartColors.green },
  { label: "OOD Detection Rate", value: "96.3%", color: chartColors.green },
  { label: "Data Drift", value: "0.012", color: chartColors.orange },
  { label: "FP Rate", value: "0.03%", color: chartColors.orange },
];

const monitorMini = [
  { label: "Temperature", value: "62°C", icon: "🌡" },
  { label: "Memory", value: "3.8 / 8.0 GB", icon: "💾" },
  { label: "Requests/min", value: "2,341", icon: "📨" },
];

export default function Intelligence() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const [activeTab, setActiveTab] = useState<Tab>("Analytics");

  return (
    <section id="intelligence" className="relative py-32 md:py-40 px-6 md:px-12 border-t border-white/[0.06]">
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 30% 20%, rgba(77,124,254,0.06) 0%, transparent 60%)",
        }}
      />
      <div className="max-w-7xl mx-auto relative z-10">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 40 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="text-center mb-12"
        >
          <span className="text-xs tracking-[0.3em] uppercase text-white/40 font-mono">
            Intelligence
          </span>
          <h2 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight">
            Analytics &{" "}
            <span className="text-gradient-blue">Model Monitoring</span>
          </h2>
          <p className="mt-4 text-white/50 max-w-xl mx-auto text-sm md:text-base">
            Real-time insights into detection performance, model health, and system metrics
          </p>
        </motion.div>

        {/* Tabs */}
        <div className="flex justify-center mb-12">
          <div className="inline-flex p-1 rounded-full bg-white/[0.04] border border-white/[0.08]">
            {tabs.map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-2.5 text-sm font-medium rounded-full transition-all duration-300 ${
                  activeTab === tab
                    ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/20"
                    : "text-white/50 hover:text-white/80"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          {activeTab === "Analytics" && (
            <div className="space-y-6">
              {/* Stat cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
                {analyticsCards.map((card, i) => (
                  <motion.div
                    key={card.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={inView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.5, delay: 0.1 * i }}
                    className="relative p-6 md:p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08] overflow-hidden group"
                  >
                    <div
                      className="absolute top-0 right-0 w-32 h-32 -translate-y-10 translate-x-10 rounded-full opacity-10"
                      style={{ background: card.color }}
                    />
                    <p className="text-xs md:text-sm text-white/40 font-mono tracking-wide mb-1.5">{card.label}</p>
                    <p className="text-3xl md:text-4xl font-extrabold text-white" style={{ color: card.color }}>
                      {card.value}
                    </p>
                  </motion.div>
                ))}
              </div>

              {/* Charts row */}
              <div className="grid md:grid-cols-3 gap-6 md:gap-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.4 }}
                  className="p-6 md:p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08]"
                >
                  <h4 className="text-sm md:text-base font-semibold text-white/70 mb-1">Detection Trends</h4>
                  <p className="text-xs text-white/30 mb-4">Last 7 days</p>
                  <div className="h-40">
                    <LineTrendChart inView={inView} />
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.5 }}
                  className="p-6 md:p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08]"
                >
                  <h4 className="text-sm md:text-base font-semibold text-white/70 mb-1">Model Performance</h4>
                  <p className="text-xs text-white/30 mb-4">Accuracy by category</p>
                  <BarChart inView={inView} />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.6 }}
                  className="p-6 md:p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08]"
                >
                  <h4 className="text-sm md:text-base font-semibold text-white/70 mb-1">Generator Type Distribution</h4>
                  <p className="text-xs text-white/30 mb-4">Deepfake generation methods</p>
                  <DonutChart inView={inView} />
                </motion.div>
              </div>
            </div>
          )}

          {activeTab === "Model Monitoring" && (
            <div className="space-y-6">
              {/* Stat cards */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 md:gap-6">
                {monitorCards.map((card, i) => (
                  <motion.div
                    key={card.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={inView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.5, delay: 0.1 * i }}
                    className="relative p-6 md:p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08] overflow-hidden group"
                  >
                    <div
                      className="absolute top-0 right-0 w-28 h-28 -translate-y-8 translate-x-8 rounded-full opacity-10"
                      style={{ background: card.color }}
                    />
                    <p className="text-xs md:text-sm text-white/40 font-mono tracking-wide mb-1.5">{card.label}</p>
                    <p className="text-3xl md:text-4xl font-extrabold" style={{ color: card.color }}>
                      {card.value}
                    </p>
                  </motion.div>
                ))}
              </div>

              {/* Charts */}
              <div className="grid md:grid-cols-2 gap-6 md:gap-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.6 }}
                  className="p-6 md:p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08]"
                >
                  <h4 className="text-sm md:text-base font-semibold text-white/70 mb-1">GPU Utilization</h4>
                  <p className="text-xs text-white/30 mb-4">Last 24 hours</p>
                  <div className="h-36">
                    <AreaChart inView={inView} />
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.7 }}
                  className="p-6 md:p-8 rounded-3xl bg-white/[0.03] border border-white/[0.08]"
                >
                  <h4 className="text-sm md:text-base font-semibold text-white/70 mb-1">Inference Latency</h4>
                  <p className="text-xs text-white/30 mb-4">ms per request (last 12 samples)</p>
                  <div className="h-28">
                    <MiniLatencyChart inView={inView} />
                  </div>
                </motion.div>
              </div>

              {/* Mini metrics */}
              <div className="grid grid-cols-3 gap-4 md:gap-6">
                {monitorMini.map((item, i) => (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={inView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.5, delay: 0.8 + i * 0.1 }}
                    className="p-5 md:p-7 rounded-3xl bg-white/[0.03] border border-white/[0.08] text-center"
                  >
                    <span className="text-2xl">{item.icon}</span>
                    <p className="text-base md:text-lg font-bold text-white mt-2">{item.value}</p>
                    <p className="text-xs text-white/40 mt-1">{item.label}</p>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </section>
  );
}
