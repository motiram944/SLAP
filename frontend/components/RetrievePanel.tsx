"use client";

import React, { useState } from "react";
import {
  Search,
  Filter,
  Sliders,
  CheckCircle2,
  XCircle,
  Cpu,
  Share2,
  Zap,
  ArrowRight,
  ShieldCheck,
  Activity,
  Layers,
} from "lucide-react";

interface RetrievePanelProps {
  apiBaseUrl: string;
}

const PRESET_QUERIES = [
  "How to authenticate with Keycloak JWT?",
  "How does Qdrant vector ANN search work?",
  "What is continuous KV cache attention prefix?",
];

export default function RetrievePanel({ apiBaseUrl }: RetrievePanelProps) {
  const [query, setQuery] = useState(PRESET_QUERIES[0]);
  const [driftThreshold, setDriftThreshold] = useState(0.75);
  const [selectedEntity, setSelectedEntity] = useState<string>("");
  const [topK, setTopK] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any | null>(null);
  const [injectedData, setInjectedData] = useState<any | null>(null);
  const [injectingId, setInjectingId] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setInjectedData(null);

    try {
      const res = await fetch(`${apiBaseUrl}/api/v1/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          drift_threshold: driftThreshold,
          query_entities: selectedEntity ? [selectedEntity] : undefined,
          top_k: topK,
          max_epistemic_variance: 0.05,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || "Retrieval failed");
      }

      const data = await res.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || "Failed to execute SLAP retrieval");
    } finally {
      setLoading(false);
    }
  };

  const handleSimulateInjection = async (match: any) => {
    setInjectingId(match.id);
    try {
      // Simulate continuous attention injection for this match vector
      const res = await fetch(`${apiBaseUrl}/api/v1/bridge/inject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vector: Array(384).fill(1.0 / Math.sqrt(384)), // Representative unit vector
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setInjectedData({ matchId: match.id, ...data });
      }
    } catch (err) {
      console.error(err);
    } finally {
      setInjectingId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Search className="w-5 h-5" />
            </span>
            Topological Pruning & Cosine Drift Retrieval
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Prunes candidates along hypergraph adjacency before ANN scoring, then filters via Cosine Drift Gate ($\tau$) and Epistemic Covariance ($\Sigma_e$).
          </p>
        </div>

        {/* Quick query chips */}
        <div className="flex flex-wrap gap-2">
          {PRESET_QUERIES.map((q, idx) => (
            <button
              key={idx}
              onClick={() => setQuery(q)}
              className="text-xs px-2.5 py-1.5 rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-emerald-300 border border-slate-700 transition-all flex items-center gap-1.5"
            >
              <Zap className="w-3 h-3 text-emerald-400" />
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Query Bar & Controls */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4">
        <div className="flex flex-col md:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search domain query (e.g. How to authenticate with Keycloak JWT?)..."
              className="w-full pl-10 pr-4 py-2.5 rounded-lg bg-slate-950 border border-slate-700/80 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 font-mono"
            />
          </div>

          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-slate-950 font-semibold text-sm shadow-lg shadow-emerald-950/50 flex items-center justify-center gap-2 transition-all disabled:opacity-50 cursor-pointer"
          >
            {loading ? (
              <span className="inline-block w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
            ) : (
              <>
                <Search className="w-4 h-4" />
                Query Manifold
              </>
            )}
          </button>
        </div>

        {/* Sliders & Filtering Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-slate-800/80 text-xs">
          {/* Drift Gate Slider */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-slate-300">
              <span className="font-medium flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-emerald-400" />
                Cosine Drift Gate ($\tau$)
              </span>
              <span className="font-mono text-emerald-300 font-semibold">{driftThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.01"
              value={driftThreshold}
              onChange={(e) => setDriftThreshold(parseFloat(e.target.value))}
              className="w-full accent-emerald-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>0.50 (Permissive)</span>
              <span>0.82 (Nominal)</span>
              <span>0.95 (Strict)</span>
            </div>
          </div>

          {/* Topological Entity Filter */}
          <div className="space-y-1">
            <label className="text-slate-300 font-medium flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-cyan-400" />
              Entity Topological Filter
            </label>
            <select
              value={selectedEntity}
              onChange={(e) => setSelectedEntity(e.target.value)}
              className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 text-slate-200 text-xs focus:outline-none focus:border-cyan-500"
            >
              <option value="">No Filter (Global Manifold)</option>
              <option value="Keycloak">Keycloak Subgraph</option>
              <option value="JWT">JWT Subgraph</option>
              <option value="Qdrant">Qdrant Subgraph</option>
              <option value="Continuous">Continuous Latent Subgraph</option>
            </select>
          </div>

          {/* Top-K Selector */}
          <div className="space-y-1">
            <label className="text-slate-300 font-medium flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-violet-400" />
              Top-K Candidate Depth: <span className="font-mono text-violet-300">{topK}</span>
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full accent-violet-400 cursor-pointer"
            />
          </div>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs">
          {error}
        </div>
      )}

      {/* Results Section */}
      {results && (
        <div className="space-y-4">
          {/* Execution Telemetry Bar */}
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
            <div className="flex items-center gap-6">
              <div>
                <span className="text-slate-500">Latency: </span>
                <span className="font-mono font-semibold text-emerald-400">
                  {results.execution_time_ms} ms
                </span>
              </div>
              <div>
                <span className="text-slate-500">Topological Pruned: </span>
                <span className="font-mono font-semibold text-cyan-400">
                  {results.topological_pruned} documents
                </span>
              </div>
              <div>
                <span className="text-slate-500">Active Candidates: </span>
                <span className="font-mono font-semibold text-white">
                  {results.total_candidates}
                </span>
              </div>
            </div>

            {/* Context Savings summary badge */}
            <div className="px-3 py-1 rounded bg-emerald-950/50 border border-emerald-800/60 text-emerald-300 font-mono text-[11px] flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-emerald-400" />
              Context Reduction: {results.token_savings.context_reduction_percent}% (
              {results.token_savings.tokens_bypassed} tokens saved)
            </div>
          </div>

          {/* Match Cards */}
          <div className="grid grid-cols-1 gap-3.5">
            {results.matches.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-xs bg-slate-900/40 rounded-lg border border-slate-800">
                No documents found matching topological or vector query constraints.
              </div>
            ) : (
              results.matches.map((match: any, idx: number) => {
                const passed = match.passed_drift_gate;
                return (
                  <div
                    key={match.id}
                    className={`p-4 rounded-xl border transition-all ${
                      passed
                        ? "bg-slate-900/90 border-slate-800 hover:border-emerald-500/50"
                        : "bg-slate-950/60 border-rose-900/40 opacity-75"
                    }`}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-slate-800/60">
                      <div className="flex items-center gap-2.5">
                        <span className="text-xs font-mono text-slate-500">#{idx + 1}</span>
                        <span className="text-xs font-mono text-slate-300">{match.id.slice(0, 8)}...</span>

                        {passed ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" />
                            Verified ($\tau \ge {driftThreshold}$)
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center gap-1">
                            <XCircle className="w-3 h-3" />
                            Drift Blocked
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-4 text-xs font-mono">
                        <div>
                          <span className="text-slate-500">Cosine Sim: </span>
                          <span
                            className={`font-bold ${
                              passed ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {match.cosine_similarity.toFixed(4)}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500">$\Sigma_e$: </span>
                          <span className="text-cyan-400">{match.epistemic_variance.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">||z||: </span>
                          <span className="text-slate-300">{match.reconstructed_norm.toFixed(4)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Text Body */}
                    <p className="text-sm text-slate-200 mt-2.5 font-sans leading-relaxed">
                      {match.text}
                    </p>

                    {/* Verdict line */}
                    <div className="mt-2 text-xs font-mono text-slate-400 flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                      <span>{match.guard_verdict}</span>
                    </div>

                    {/* Hypergraph Edge Tags & Action */}
                    <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-[10px] uppercase text-slate-500 font-semibold mr-1">
                          Adjacency:
                        </span>
                        {match.edges.slice(0, 3).map((edge: any, eIdx: number) => (
                          <span
                            key={eIdx}
                            className="text-[10px] px-2 py-0.5 rounded bg-slate-950 border border-slate-800 font-mono text-slate-400"
                          >
                            {edge.source} &rarr; {edge.relation} &rarr; {edge.target}
                          </span>
                        ))}
                      </div>

                      {passed && (
                        <button
                          onClick={() => handleSimulateInjection(match)}
                          disabled={injectingId === match.id}
                          className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer"
                        >
                          {injectingId === match.id ? (
                            <span className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                          ) : (
                            <Cpu className="w-3.5 h-3.5" />
                          )}
                          Inject KV Prefix
                        </button>
                      )}
                    </div>

                    {/* Injected Preview Sub-Card */}
                    {injectedData && injectedData.matchId === match.id && (
                      <div className="mt-3 p-3 rounded-lg bg-cyan-950/30 border border-cyan-800/40 text-xs font-mono space-y-1.5">
                        <div className="text-cyan-300 font-semibold flex items-center justify-between">
                          <span>Virtual Prefix Injected into LLM Attention Cache</span>
                          <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-900/60 text-cyan-200">
                            Dim: {injectedData.virtual_token_dimension} (4096-d)
                          </span>
                        </div>
                        <div className="text-[11px] text-slate-400">
                          Status: {injectedData.kv_cache_injection.status} &bull; Attention Heads:{" "}
                          {injectedData.kv_cache_injection.num_attention_heads} &bull; Head Dim:{" "}
                          {injectedData.kv_cache_injection.head_dimension}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
