"use client";

import React, { useState } from "react";
import {
  Layers,
  Database,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  Zap,
  ArrowRight,
  Sparkles,
  Copy,
  Check,
} from "lucide-react";

interface IngestPanelProps {
  onIngestSuccess?: () => void;
  apiBaseUrl: string;
}

const TEMPLATES = [
  {
    title: "Keycloak JWT Bearer Token",
    text: "User authentication requires a Keycloak JWT bearer token with RS256 signatures to authorize downstream microservice requests.",
    variance: 0.012,
  },
  {
    title: "Qdrant S^{d-1} Hypersphere ANN",
    text: "The Qdrant vector database indexes S^{d-1} continuous hypersphere coordinates using Cosine distance for sub-millisecond ANN search.",
    variance: 0.018,
  },
  {
    title: "Continuous KV Cache Soft-Prompt",
    text: "Continuous KV cache injection prepends virtual soft-prompt tokens directly to LLM attention, bypassing discrete text serialization.",
    variance: 0.015,
  },
];

export default function IngestPanel({ onIngestSuccess, apiBaseUrl }: IngestPanelProps) {
  const [text, setText] = useState(TEMPLATES[0].text);
  const [variance, setVariance] = useState(TEMPLATES[0].variance);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<any | null>(null);
  const [copied, setCopied] = useState(false);

  const handleIngest = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${apiBaseUrl}/api/v1/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text.trim(),
          epistemic_variance: variance,
          metadata: {
            source: "SLAP Studio Playground",
            timestamp: new Date().toISOString(),
          },
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || "Ingest failed");
      }

      const data = await res.json();
      setResponse(data);
      if (onIngestSuccess) onIngestSuccess();
    } catch (err: any) {
      setError(err.message || "Failed to ingest into SLAP manifold");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!response) return;
    navigator.clipboard.writeText(JSON.stringify(response.envelope, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Layers className="w-5 h-5" />
            </span>
            Continuous Ingestion & S^{"{d-1}"} Projection
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Projects hidden states through orthogonal matrix W_ortho, quantizes residual delta r &rarr; INT8, and builds topological hyperedges.
          </p>

        </div>

        {/* Template Quick Select */}
        <div className="flex flex-wrap gap-2">
          {TEMPLATES.map((tmpl, idx) => (
            <button
              key={idx}
              onClick={() => {
                setText(tmpl.text);
                setVariance(tmpl.variance);
              }}
              className="text-xs px-2.5 py-1.5 rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-cyan-300 border border-slate-700 transition-all flex items-center gap-1.5"
            >
              <Sparkles className="w-3 h-3 text-cyan-400" />
              {tmpl.title}
            </button>
          ))}
        </div>
      </div>

      {/* Main Input Form */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
              Domain Text Payload
            </label>
            <textarea
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste domain text to project onto continuous manifold..."
              className="w-full px-3.5 py-2.5 rounded-lg bg-slate-900/90 border border-slate-700/80 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all font-mono"
            />
          </div>

          {/* Epistemic Variance Slider */}
          <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div>
              <div className="text-xs font-medium text-slate-300 flex items-center gap-2">
                Epistemic Uncertainty / Covariance ($\Sigma_e$)
                <span className="text-xs px-2 py-0.5 rounded font-mono bg-cyan-950/80 text-cyan-300 border border-cyan-800/50">
                  {variance.toFixed(4)}
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Model confidence metric. WAF blocks values exceeding $\Sigma_e &gt; 0.050$.
              </p>
            </div>
            <input
              type="range"
              min="0.005"
              max="0.070"
              step="0.001"
              value={variance}
              onChange={(e) => setVariance(parseFloat(e.target.value))}
              className="w-full md:w-44 accent-cyan-400 cursor-pointer"
            />
          </div>

          {/* Ingest Action Button */}
          <button
            onClick={handleIngest}
            disabled={loading || !text.trim()}
            className="w-full py-3 rounded-lg bg-gradient-to-r from-cyan-600 via-cyan-500 to-emerald-500 hover:from-cyan-500 hover:to-emerald-400 text-slate-950 font-semibold text-sm shadow-lg shadow-cyan-950/50 flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {loading ? (
              <span className="inline-block w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
            ) : (
              <>
                <Zap className="w-4 h-4 fill-slate-950" />
                Project, Quantize & Index into SLAP Store
              </>
            )}
          </button>

          {error && (
            <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold">Ingestion Rejection: </span>
                {error}
              </div>
            </div>
          )}
        </div>

        {/* Live Compression & Projection Feedback */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              Continuous Algebraic Telemetry
            </h3>

            {response ? (
              <div className="space-y-3">
                {/* Metric Grid */}
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase">Unit Norm ||z||_2</div>
                    <div className="text-sm font-mono font-bold text-emerald-400 flex items-center gap-1 mt-0.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      1.000000
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase">Residual SNR</div>
                    <div className="text-sm font-mono font-bold text-cyan-400 mt-0.5">
                      {response.compression_stats.snr_db.toFixed(1)} dB
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase">Quantization Mode</div>
                    <div className="text-sm font-mono font-bold text-violet-400 mt-0.5">
                      INT8 (Affine)
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase">Compression Ratio</div>
                    <div className="text-sm font-mono font-bold text-emerald-400 mt-0.5">
                      {response.compression_stats.compression_ratio}x
                    </div>
                  </div>
                </div>

                {/* Token Savings Card */}
                <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-800/40">
                  <div className="flex items-center justify-between text-xs font-semibold text-cyan-300">
                    <span>Virtual Token Attention Prefix</span>
                    <span className="font-mono text-emerald-400">
                      -{response.token_savings.context_reduction_percent}% Context
                    </span>
                  </div>
                  <div className="mt-2 text-[11px] text-slate-300 space-y-1">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Tokens Bypassed:</span>
                      <span className="font-mono font-medium text-white">
                        {response.token_savings.tokens_bypassed} tokens
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">KV Cache RAM Saved:</span>
                      <span className="font-mono font-medium text-white">
                        {response.token_savings.saved_kv_cache_mb} MB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">TTFT Acceleration:</span>
                      <span className="font-mono font-medium text-emerald-300">
                        {response.token_savings.estimated_ttft_speedup}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Hyperedges Extracted */}
                <div>
                  <div className="text-[11px] font-semibold text-slate-400 mb-1.5 flex items-center justify-between">
                    <span>Hypergraph Topology ({response.envelope.edges.length} edges)</span>
                    <span className="text-[10px] text-cyan-400 font-mono">Qdrant Indexed</span>
                  </div>
                  <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                    {response.envelope.edges.map((e: any, idx: number) => (
                      <div
                        key={idx}
                        className="text-[11px] px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800/80 font-mono flex items-center gap-1.5 text-slate-300"
                      >
                        <span className="text-cyan-400 font-medium">{e.source}</span>
                        <ArrowRight className="w-3 h-3 text-slate-500" />
                        <span className="text-amber-400">{e.relation}</span>
                        <ArrowRight className="w-3 h-3 text-slate-500" />
                        <span className="text-emerald-400 font-medium">{e.target}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* SlapEnvelope JSON copy */}
                <button
                  onClick={handleCopy}
                  className="w-full text-xs py-1.5 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      Envelope Copied to Clipboard
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      Copy Wire SlapEnvelope (Protobuf Map)
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500 text-xs">
                Submit an ingestion payload to observe real-time S^{"{d-1}"} orthogonal projection and residual quantization.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
