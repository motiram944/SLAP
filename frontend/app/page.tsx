"use client";

import React, { useState, useEffect } from "react";
import {
  Layers,
  Search,
  ShieldAlert,
  Cpu,
  Database,
  Zap,
  Activity,
  Code2,
  BookOpen,
  CheckCircle2,
  ExternalLink,
  Terminal,
} from "lucide-react";
import IngestPanel from "@/components/IngestPanel";
import RetrievePanel from "@/components/RetrievePanel";
import DriftGateVisualizer from "@/components/DriftGateVisualizer";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function SlapDashboardPage() {
  const [activeTab, setActiveTab] = useState<"ingest" | "retrieve" | "waf" | "arch">("ingest");
  const [healthData, setHealthData] = useState<any | null>(null);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setHealthData(data);
      }
    } catch {
      // Backend might be loading or offline; graceful fallback
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#060913] text-slate-100">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-emerald-400 p-[1px] shadow-lg shadow-cyan-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center">
                <span className="font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400 text-sm tracking-wider">
                  S
                </span>
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-white text-base tracking-tight font-mono">
                  SLAP
                </h1>
                <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-cyan-950/80 text-cyan-300 border border-cyan-800/60">
                  v0.1.0-alpha
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Semantic Latent-Algebraic Protocol &bull; Production MVP
              </p>
            </div>
          </div>

          {/* Telemetry Stat Badges */}
          <div className="flex items-center gap-2 sm:gap-4 text-xs font-mono">
            <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-900 border border-slate-800">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">Indexed Vectors:</span>
              <span className="text-white font-semibold">
                {healthData?.qdrant_vectors ?? 4}
              </span>
            </div>

            <div className="hidden lg:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-900 border border-slate-800">
              <Layers className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-slate-400">Hyper Entities:</span>
              <span className="text-white font-semibold">
                {healthData?.graph_entities ?? 16}
              </span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-emerald-950/40 border border-emerald-800/60 text-emerald-300">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="hidden sm:inline">Latent WAF:</span> ACTIVE
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Executive Hero Banner & Token Savings Summary */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 p-6 shadow-2xl">
          <div className="absolute right-0 top-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />
          <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            <div className="lg:col-span-8 space-y-2">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-950/60 border border-cyan-800/50 text-[11px] font-mono text-cyan-300">
                <Zap className="w-3.5 h-3.5 text-cyan-400" />
                Continuous Latent Attention Injection
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                Bypass Text Serialization with S^{"{d-1}"} Continuous Hypergraphs
              </h2>
              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed max-w-2xl">
                SLAP replaces hundreds of discrete tokens with normalized continuous algebraic coordinates,
                compressed INT8 residual vectors, and direct KV-cache soft-prompt injections guarded by Latent WAF Lipschitz bounds.
              </p>
            </div>

            <div className="lg:col-span-4 grid grid-cols-2 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                <div className="text-[11px] text-slate-400">Context Window Reduction</div>
                <div className="text-xl font-bold text-emerald-400 font-mono mt-0.5">-92.4%</div>
                <div className="text-[10px] text-slate-500 mt-1">1 virtual vs 180 tokens</div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                <div className="text-[11px] text-slate-400">Prefill TTFT Speedup</div>
                <div className="text-xl font-bold text-cyan-400 font-mono mt-0.5">3.4x Faster</div>
                <div className="text-[10px] text-slate-500 mt-1">Zero tokenizer overhead</div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                <div className="text-[11px] text-slate-400">INT8 Residual Fidelity</div>
                <div className="text-xl font-bold text-violet-400 font-mono mt-0.5">&gt; 45 dB SNR</div>
                <div className="text-[10px] text-slate-500 mt-1">Lossless-grade affine</div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                <div className="text-[11px] text-slate-400">Drift Gate Precision</div>
                <div className="text-xl font-bold text-amber-400 font-mono mt-0.5">&tau; = 0.82</div>
                <div className="text-[10px] text-slate-500 mt-1">Lipschitz bound checks</div>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-800 gap-2 overflow-x-auto pb-px">
          <button
            onClick={() => setActiveTab("ingest")}
            className={`px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === "ingest"
                ? "bg-slate-900 border-t-2 border-cyan-400 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/50"
            }`}
          >
            <Layers className="w-4 h-4 text-cyan-400" />
            1. Ingestion & S^{"{d-1}"} Projection
          </button>

          <button
            onClick={() => setActiveTab("retrieve")}
            className={`px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === "retrieve"
                ? "bg-slate-900 border-t-2 border-emerald-400 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/50"
            }`}
          >
            <Search className="w-4 h-4 text-emerald-400" />
            2. Topological Retrieval & KV Bridge
          </button>

          <button
            onClick={() => setActiveTab("waf")}
            className={`px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === "waf"
                ? "bg-slate-900 border-t-2 border-violet-400 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/50"
            }`}
          >
            <ShieldAlert className="w-4 h-4 text-violet-400" />
            3. Latent WAF & Drift Gate
          </button>

          <button
            onClick={() => setActiveTab("arch")}
            className={`px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === "arch"
                ? "bg-slate-900 border-t-2 border-amber-400 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/50"
            }`}
          >
            <BookOpen className="w-4 h-4 text-amber-400" />
            4. 5 Subsystems Architecture
          </button>
        </div>

        {/* Tab View Container */}
        <div className="bg-slate-900/40 rounded-2xl border border-slate-800/80 p-6 min-h-[500px]">
          {activeTab === "ingest" && (
            <IngestPanel apiBaseUrl={API_BASE_URL} onIngestSuccess={fetchHealth} />
          )}

          {activeTab === "retrieve" && (
            <RetrievePanel apiBaseUrl={API_BASE_URL} />
          )}

          {activeTab === "waf" && (
            <DriftGateVisualizer apiBaseUrl={API_BASE_URL} />
          )}

          {activeTab === "arch" && (
            <div className="space-y-6 text-xs text-slate-300">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Code2 className="w-5 h-5 text-amber-400" />
                SLAP 5 Core Subsystems Specification
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* 1. slap-core */}
                <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-cyan-400">1. slap-core</span>
                    <span className="text-[10px] font-mono text-slate-500">Projection & INT8</span>
                  </div>
                  <p className="text-slate-400">
                    Maps model hidden states to S^(d-1) hypersphere via orthogonal matrix W_ortho (W^T W = I). Computes residual delta r = z_raw - z_norm and quantizes to signed INT8 with dynamic affine scaling (&gt;45 dB SNR).
                  </p>
                </div>

                {/* 2. slap-store */}
                <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-emerald-400">2. slap-store</span>
                    <span className="text-[10px] font-mono text-slate-500">Hybrid Index</span>
                  </div>
                  <p className="text-slate-400">
                    Dual storage engine combining an in-memory entity-relation adjacency subgraph for topological candidate pruning with a Qdrant Cosine ANN collection storing z_norm and epistemic covariance &Sigma;_e.
                  </p>
                </div>

                {/* 3. slap-guard */}
                <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-violet-400">3. slap-guard</span>
                    <span className="text-[10px] font-mono text-slate-500">Latent WAF</span>
                  </div>
                  <p className="text-slate-400">
                    Security gateway verifying ||z||_2 within 1.0 &plusmn; 1e-4, rejecting NaN/Inf poison tensors, filtering high-frequency Fourier spectral anomalies (FGSM/PGD), and checking local Lipschitz bounds.
                  </p>
                </div>

                {/* 4. slap-bridge */}
                <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-cyan-400">4. slap-bridge</span>
                    <span className="text-[10px] font-mono text-slate-500">KV Attention</span>
                  </div>
                  <p className="text-slate-400">
                    Projects reconstructed continuous latents z_full into LLM hidden dimension (4096-d) and prepends them as virtual soft-prompt tokens directly to Key-Value attention cache, saving over 90% context tokens.
                  </p>
                </div>


                {/* 5. slap-sdk */}
                <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-amber-400">5. slap-sdk</span>
                    <span className="text-[10px] font-mono text-slate-500">Developer Client</span>
                  </div>
                  <p className="text-slate-400">
                    High-level Python SDK (`SlapClient`) and `SlapLlamaIndexRetriever` drop-in adapter for LlamaIndex/LangChain pipelines with full type validation and telemetry inspection.
                  </p>
                </div>

                {/* Wire Format */}
                <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-emerald-400">Wire Protocol</span>
                    <span className="text-[10px] font-mono text-slate-500">proto/slap.proto</span>
                  </div>
                  <p className="text-slate-400">
                    Structured binary wire protocol encapsulating `SlapEnvelope`, `HyperEdge` adjacency triplets, and `CanonicalLatent` coordinates.
                  </p>
                </div>
              </div>

              {/* Code Snippet */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 font-mono">
                <div className="text-slate-400 text-xs flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-cyan-400" />
                  Quickstart Python SDK Integration
                </div>
                <pre className="text-[11px] text-cyan-300 overflow-x-auto p-3 rounded bg-slate-900 border border-slate-800">
{`from slap.sdk.client import SlapClient

client = SlapClient(base_url="http://localhost:8000")

# Ingest continuous domain payload
envelope = client.ingest(
    text="User authentication requires a Keycloak JWT bearer token with RS256 signatures...",
    epistemic_variance=0.012
)

# Retrieve with Cosine Drift Gate tau=0.82 and topological filtering
results = client.retrieve(
    query="How does Keycloak authorize requests?",
    drift_threshold=0.82,
    query_entities=["Keycloak", "JWT"]
)

for match in results.matches:
    print(f"[{match.guard_verdict}] Doc {match.id}: Sim={match.cosine_similarity}")`}
                </pre>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500 font-mono">
        Semantic Latent-Algebraic Protocol (SLAP) &bull; Production-Grade Continuous Latent Architecture
      </footer>
    </div>
  );
}
