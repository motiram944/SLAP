"use client";

import React, { useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  AlertOctagon,
  Radio,
  Sliders,
  CheckCircle,
  XCircle,
  HelpCircle,
  Zap,
} from "lucide-react";

interface DriftGateVisualizerProps {
  apiBaseUrl: string;
}

interface TestPreset {
  id: string;
  title: string;
  description: string;
  variance: number;
  expectedResult: "PASSED" | "BLOCKED";
  vectorGenerator: () => number[];
}

const PRESETS: TestPreset[] = [
  {
    id: "clean",
    title: "Unit S^{d-1} Hypersphere Vector",
    description: "Conforms to unit hypersphere manifold with low epistemic variance.",
    variance: 0.015,
    expectedResult: "PASSED",
    vectorGenerator: () => {
      // Deterministic smooth continuous vector normalized to exactly 1.0
      const raw = Array.from({ length: 384 }, (_, i) => Math.sin(i * 0.08) + 0.5 * Math.cos(i * 0.13));
      const norm = Math.sqrt(raw.reduce((acc, val) => acc + val * val, 0));
      return raw.map((v) => v / norm);
    },
  },
  {
    id: "norm_violation",
    title: "Manifold Boundary Expansion (+4% Norm)",
    description: "Vector length ||z||_2 = 1.040 deviates beyond 1e-4 tolerance band.",
    variance: 0.018,
    expectedResult: "BLOCKED",
    vectorGenerator: () => {
      const raw = Array.from({ length: 384 }, (_, i) => Math.sin(i * 0.08));
      const norm = Math.sqrt(raw.reduce((acc, val) => acc + val * val, 0));
      return raw.map((v) => (v / norm) * 1.04);
    },
  },
  {
    id: "spectral_noise",
    title: "High-Frequency Spectral Injection (FGSM/PGD)",
    description: "Adversarial noise concentrates energy in top Fourier spectrum bins.",
    variance: 0.012,
    expectedResult: "BLOCKED",
    vectorGenerator: () => {
      // High-frequency alternating sign attack (-1)^i
      const raw = Array.from({ length: 384 }, (_, i) => (i % 2 === 0 ? 1.0 : -1.0));
      const norm = Math.sqrt(raw.reduce((acc, val) => acc + val * val, 0));
      return raw.map((v) => v / norm);
    },
  },
  {
    id: "variance_breach",
    title: "Excessive Epistemic Variance (Σ_e = 0.085)",
    description: "Out-of-distribution uncertainty exceeds safety bound Σ_e > 0.050.",
    variance: 0.085,
    expectedResult: "BLOCKED",
    vectorGenerator: () => {
      const raw = Array.from({ length: 384 }, (_, i) => Math.cos(i * 0.05));
      const norm = Math.sqrt(raw.reduce((acc, val) => acc + val * val, 0));
      return raw.map((v) => v / norm);
    },
  },
  {
    id: "nan_poison",
    title: "NaN Floating-Point Gradient Bomb",
    description: "Adversarial payload injects NaN/Inf coordinates to poison attention.",
    variance: 0.010,
    expectedResult: "BLOCKED",
    vectorGenerator: () => {
      const raw = Array.from({ length: 384 }, () => 0.05);
      raw[42] = NaN;
      return raw;
    },
  },
];

export default function DriftGateVisualizer({ apiBaseUrl }: DriftGateVisualizerProps) {
  const [activePreset, setActivePreset] = useState<TestPreset>(PRESETS[0]);
  const [customVariance, setCustomVariance] = useState<number>(PRESETS[0].variance);
  const [inspecting, setInspecting] = useState(false);
  const [verdict, setVerdict] = useState<any | null>(null);

  const runWafInspection = async (preset: TestPreset, varianceVal: number) => {
    setInspecting(true);
    setVerdict(null);

    try {
      const vec = preset.vectorGenerator();
      const res = await fetch(`${apiBaseUrl}/api/v1/guard/inspect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vector: vec,
          epistemic_variance: varianceVal,
          drift_threshold: 0.82,
          max_variance: 0.05,
        }),
      });

      const data = await res.json();
      setVerdict(data);
    } catch (err: any) {
      setVerdict({
        passed: false,
        reason: `Network/Inspection Error: ${err.message}`,
        l2_norm: 0,
        norm_error: 1.0,
        spectral_energy: 0,
        epistemic_variance: varianceVal,
        lipschitz_ratio: 0,
      });
    } finally {
      setInspecting(false);
    }
  };

  const handleSelectPreset = (preset: TestPreset) => {
    setActivePreset(preset);
    setCustomVariance(preset.variance);
    runWafInspection(preset, preset.variance);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="p-1.5 rounded-lg bg-violet-500/10 text-violet-400 border border-violet-500/20">
            <ShieldAlert className="w-5 h-5" />
          </span>
          Latent WAF Security Gateway & Adversarial Inspector
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Enforces S^{"{d-1}"} manifold boundaries, Lipschitz continuity, FFT spectral anomaly bounds, and epistemic covariance ($\Sigma_e$).
        </p>
      </div>

      {/* Preset Attacks Matrix */}
      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
          Adversarial Test Vectors & Integrity Vectors
        </label>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-2.5">
          {PRESETS.map((p) => {
            const isSelected = activePreset.id === p.id;
            return (
              <button
                key={p.id}
                onClick={() => handleSelectPreset(p)}
                className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                  isSelected
                    ? "bg-slate-800/90 border-cyan-500 text-white shadow-lg shadow-cyan-950/40"
                    : "bg-slate-900/60 border-slate-800 hover:border-slate-700 text-slate-300"
                }`}
              >
                <div className="flex items-center justify-between text-xs font-semibold mb-1">
                  <span className="truncate pr-1">{p.title}</span>
                  {p.expectedResult === "PASSED" ? (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      PASS
                    </span>
                  ) : (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      BLOCK
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 leading-tight">{p.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Manual Trigger & Sliders */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex-1 max-w-md">
          <div className="flex justify-between text-xs text-slate-300 mb-1">
            <span>Epistemic Uncertainty ($\Sigma_e$):</span>
            <span className="font-mono text-cyan-400 font-semibold">{customVariance.toFixed(4)}</span>
          </div>
          <input
            type="range"
            min="0.005"
            max="0.090"
            step="0.002"
            value={customVariance}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setCustomVariance(val);
              runWafInspection(activePreset, val);
            }}
            className="w-full accent-cyan-400 cursor-pointer"
          />
        </div>

        <button
          onClick={() => runWafInspection(activePreset, customVariance)}
          disabled={inspecting}
          className="px-5 py-2.5 rounded-lg bg-gradient-to-r from-violet-600 to-cyan-600 hover:from-violet-500 hover:to-cyan-500 text-white font-semibold text-xs shadow-md transition-all flex items-center gap-2 cursor-pointer"
        >
          {inspecting ? (
            <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <Activity className="w-3.5 h-3.5" />
          )}
          Re-evaluate Latent Guard
        </button>
      </div>

      {/* Live Verdict & Telemetry Display */}
      {verdict && (
        <div className="space-y-4">
          {/* Main Verdict Banner */}
          <div
            className={`p-4 rounded-xl border flex items-start gap-3.5 ${
              verdict.passed
                ? "bg-emerald-950/30 border-emerald-800/60 text-emerald-200"
                : "bg-rose-950/40 border-rose-800/60 text-rose-200"
            }`}
          >
            {verdict.passed ? (
              <ShieldCheck className="w-6 h-6 text-emerald-400 flex-shrink-0 mt-0.5" />
            ) : (
              <ShieldAlert className="w-6 h-6 text-rose-400 flex-shrink-0 mt-0.5" />
            )}
            <div>
              <div className="text-sm font-bold flex items-center gap-2">
                <span>{verdict.passed ? "ADMISSIBLE: PASSED ALL LATENT GATES" : "REJECTED BY LATENT WAF GATEWAY"}</span>
                <span
                  className={`text-[10px] px-2 py-0.5 rounded font-mono uppercase ${
                    verdict.passed
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-rose-500/20 text-rose-300"
                  }`}
                >
                  {verdict.passed ? "Verified" : "Security Block"}
                </span>
              </div>
              <p className="text-xs text-slate-300 font-mono mt-1">{verdict.reason}</p>
            </div>
          </div>

          {/* Mathematical Gauge Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            {/* L2 Norm Gauge */}
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Hypersphere ||z||_2</span>
                <span className="text-[10px] font-mono text-slate-500">Target: 1.0 &plusmn; 1e-4</span>
              </div>
              <div
                className={`text-lg font-mono font-bold ${
                  verdict.norm_error <= 1e-4 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {verdict.l2_norm.toFixed(6)}
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Norm Error: &Delta; = {verdict.norm_error.toExponential(2)}
              </div>
            </div>

            {/* Spectral High-Frequency Ratio */}
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Spectral Energy FFT</span>
                <span className="text-[10px] font-mono text-slate-500">Limit: &lt; 0.65</span>
              </div>
              <div
                className={`text-lg font-mono font-bold ${
                  !verdict.is_spectral_anomaly ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {(verdict.spectral_energy * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Status: {verdict.is_spectral_anomaly ? "Anomalous Peak" : "Conforming Spectrum"}
              </div>
            </div>

            {/* Lipschitz Gradient Ratio */}
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Lipschitz Bound</span>
                <span className="text-[10px] font-mono text-slate-500">Bound: &lt; 4.5</span>
              </div>
              <div
                className={`text-lg font-mono font-bold ${
                  verdict.lipschitz_ratio <= 4.5 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {verdict.lipschitz_ratio.toFixed(2)}x
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Manifold Gradient Roughness
              </div>
            </div>

            {/* Epistemic Covariance */}
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Epistemic Covariance &Sigma;_e</span>
                <span className="text-[10px] font-mono text-slate-500">Max: 0.050</span>
              </div>
              <div
                className={`text-lg font-mono font-bold ${
                  verdict.epistemic_variance <= 0.05 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {verdict.epistemic_variance.toFixed(4)}
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Model Uncertainty Boundary
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
