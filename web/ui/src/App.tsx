import { useEffect, useMemo, useState } from "react";
import type {
  MatchEntry,
  MatchEvaluation,
  RecommendedSummaryEntry,
  SafeSummaryEntry,
  RunDetail,
  RunSummary,
  WatcherState,
} from "./types";
import { PivotChart } from "./components/PivotChart";
import { PivotOddsChart } from "./components/PivotOddsChart";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const WATCHER_STALE_THRESHOLD_MS = 3 * 60 * 1000; // 3 minutes

type RunsResponse = {
  runs: RunSummary[];
  total: number;
};

const formatDate = (value?: string) => {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch (error) {
    return value;
  }
};

const heartbeatAge = (heartbeat?: string) => {
  if (!heartbeat) return null;
  const ts = Date.parse(heartbeat);
  if (Number.isNaN(ts)) return null;
  return Date.now() - ts;
};

const watcherClass = (watcher?: WatcherState | null) => {
  if (!watcher) return "status status--unknown";
  if (watcher.state !== "running") return "status status--error";
  const age = heartbeatAge(watcher.heartbeat);
  if (age !== null && age > WATCHER_STALE_THRESHOLD_MS) {
    return "status status--warning";
  }
  return "status status--ok";
};

type BestEvaluation = {
  pivot: number;
  edge: number;
  side: "over" | "under";
  odds?: number;
  label?: string;
  probability?: number;
  kelly?: number;
} | null;

const pickBestEvaluation = (match: MatchEntry, summaryEntry?: RecommendedSummaryEntry): BestEvaluation | null => {
  const evaluations = match.evaluations || [];

  const extractKellyFromOffer = (
    source: MatchEvaluation["over"] | MatchEvaluation["under"] | undefined,
  ): number | undefined => {
    if (!source?.kelly) return undefined;
    const kellyDetail = source.kelly;
    const anyKelly = kellyDetail as Record<string, unknown>;
    if (typeof anyKelly["0.66x"] === "number") {
      return anyKelly["0.66x"] as number;
    }
    if (kellyDetail.scaled && typeof kellyDetail.scaled["0.66x"] === "number") {
      return kellyDetail.scaled["0.66x"];
    }
    if (kellyDetail.scaled) {
      const first = Object.values(kellyDetail.scaled)[0];
      if (typeof first === "number") return first;
    }
    if (typeof kellyDetail.full === "number") {
      return kellyDetail.full;
    }
    return undefined;
  };

  const resolveOddsFromMarkets = (pivot: number | undefined, side: "over" | "under") => {
    if (!match.match?.markets || pivot === undefined) return undefined;
    const pivotNum = Number(pivot);
    const candidate = match.match.markets.find((market) => {
      if (typeof market.pivot !== "number") return false;
      return Math.abs(market.pivot - pivotNum) < 1e-6;
    });
    if (!candidate) return undefined;
    return side === "over" ? candidate.over_odds ?? undefined : candidate.under_odds ?? undefined;
  };

  const matchOfferFromSummary = (entry: RecommendedSummaryEntry | undefined): BestEvaluation | null => {
    if (!entry) return null;
    const pivot = entry.pivot ?? Number.NaN;
    const label = entry.label;
    const side = (entry.side as "over" | "under") ?? "over";
    const labelLower = label?.toLowerCase();
    const evaluation = evaluations.find((candidate) => {
      const pivotMatch =
        typeof candidate.pivot === "number" && typeof entry.pivot === "number"
          ? Math.abs(candidate.pivot - entry.pivot) < 1e-6
          : false;
      const labelMatch = labelLower && candidate.label?.toLowerCase() === labelLower;
      return (pivotMatch || labelMatch) && Boolean(candidate[side]);
    });
    const offer = evaluation ? evaluation[side] : undefined;
    const odds = offer?.odds ?? resolveOddsFromMarkets(entry.pivot, side);
    const probability = entry.probability ?? offer?.probability;
    const kelly =
      typeof entry.kelly === "number" ? entry.kelly : extractKellyFromOffer(offer);
    return {
      pivot,
      edge: entry.edge ?? offer?.edge ?? 0,
      side,
      odds,
      label: label ?? evaluation?.label,
      probability,
      kelly,
    };
  };

  if (summaryEntry) {
    const fromSummary = matchOfferFromSummary(summaryEntry);
    if (fromSummary) {
      return fromSummary;
    }
  } else if (match.summary?.recommended?.length) {
    const fromSummary = matchOfferFromSummary(match.summary.recommended[0]);
    if (fromSummary) {
      return fromSummary;
    }
  }

  if (summaryEntry) {
    return null;
  }

  return null;
};

const App = () => {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunDetail | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [watcher, setWatcher] = useState<WatcherState | null>(null);

  const fetchRuns = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/runs?limit=20`);
      if (!response.ok) throw new Error(`Erreur API /runs: ${response.status}`);
      const payload = (await response.json()) as RunsResponse;
      setRuns(payload.runs);
      setError(null);
    } catch (err) {
      console.error(err);
      setError((err as Error).message);
    }
  };

  const fetchRunDetails = async (runId: string) => {
    setLoadingRun(true);
    try {
      const response = await fetch(`${API_BASE_URL}/runs/${runId}`);
      if (!response.ok) throw new Error(`Erreur API /runs/${runId}: ${response.status}`);
      const payload = (await response.json()) as RunDetail;
      setSelectedRun(payload);
      setError(null);
    } catch (err) {
      console.error(err);
      setError((err as Error).message);
    } finally {
      setLoadingRun(false);
    }
  };

  const fetchWatcher = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/watcher-state`);
      if (!response.ok) throw new Error("Watcher state introuvable");
      const payload = (await response.json()) as WatcherState;
      setWatcher(payload);
    } catch (err) {
      console.warn(err);
    }
  };

  useEffect(() => {
    fetchRuns();
    fetchWatcher();
    const interval = setInterval(fetchWatcher, 30000);
    return () => clearInterval(interval);
  }, []);

  const recommendedList = useMemo(() => {
    if (!selectedRun) return [];
    const entries: (BestEvaluation & { matchIndex: number })[] = [];
    selectedRun.matches?.forEach((match, matchIndex) => {
      const list = match.summary?.recommended || [];
      list.forEach((entry) => {
        const enriched = pickBestEvaluation(match, entry);
        if (enriched) {
          entries.push({ ...enriched, matchIndex });
        }
      });
    });
    return entries;
  }, [selectedRun]);

  const safeList = useMemo(() => {
    if (!selectedRun) return [];
    const entries: Array<{ pick: SafeSummaryEntry; matchLabel?: string }> = [];
    selectedRun.matches?.forEach((match) => {
      const safe = match.summary?.safe;
      if (!safe) return;
      let safePick: SafeSummaryEntry | undefined | null = safe.pick;
      if (!safePick) {
        const legacy = (["over", "under"] as const)
          .map((key) => safe[key])
          .find((entry) => entry);
        if (legacy) {
          safePick = legacy;
        } else if (typeof safe.paired === "object" && safe.paired) {
          safePick = safe.paired.over ?? safe.paired.under ?? null;
        }
      }
      if (!safePick) return;
      const home = match.match?.home?.name;
      const away = match.match?.away?.name;
      const matchLabel = home && away ? `${away} @ ${home}` : undefined;
      entries.push({ pick: safePick, matchLabel });
    });
    return entries;
  }, [selectedRun]);

  const safeReason = useMemo(() => {
    if (!selectedRun) return null;
    for (const match of selectedRun.matches || []) {
      const safe = match.summary?.safe;
      const reason =
        safe?.reason ||
        (typeof safe?.paired === "object" ? safe?.paired?.reason : undefined);
      if (reason) return reason;
    }
    return null;
  }, [selectedRun]);

  return (
    <div className="layout">
      <header>
        <h1>NBA Mass Prediction Viewer</h1>
        <div className={watcherClass(watcher)}>
          <strong>Watcher:</strong> {watcher?.state ?? "unknown"}
          <span>
            Dernier heartbeat: {formatDate(watcher?.heartbeat)}
          </span>
          <span>PID: {watcher?.pid ?? "—"}</span>
          <span>Total processés: {watcher?.processed_total ?? 0}</span>
          {watcher?.last_error && (
            <span className="status__error">Dernière erreur: {watcher.last_error.message}</span>
          )}
        </div>
      </header>

      {error && <div className="alert alert--error">{error}</div>}

      <main>
        <section className="panel">
          <div className="panel__header">
            <h2>Runs récents</h2>
            <button onClick={fetchRuns}>Rafraîchir</button>
          </div>
          <div className="run-list">
            {(() => {
              const groups: Record<string, RunSummary[]> = {};
              runs.forEach((run) => {
                const dateKey =
                  run.created_at?.split("T")[0] ||
                  (run.path ? run.path.split("/")[0] : "Date inconnue");
                if (!groups[dateKey]) {
                  groups[dateKey] = [];
                }
                groups[dateKey].push(run);
              });
              const orderedDates = Object.keys(groups).sort((a, b) => (a > b ? -1 : 1));
              return orderedDates.map((dateKey) => {
                const displayDate = new Date(dateKey).toLocaleDateString();
                return (
                  <div key={dateKey} className="run-group">
                    <div className="run-group__label">{displayDate}</div>
                    {groups[dateKey].map((run) => (
                      <article
                        key={run.run_id}
                        className={`run-card ${selectedRun?.run_id === run.run_id ? "run-card--active" : ""}`}
                        onClick={() => fetchRunDetails(run.run_id)}
                      >
                        <h3>{run.run_id}</h3>
                        <p>{formatDate(run.created_at)}</p>
                        <p>Matches: {run.matches_count}</p>
                        <p>{run.model_label}</p>
                        {run.best_bet && (
                          <p className="run-card__edge">
                            Edge {(run.best_bet.edge * 100).toFixed(2)}% – {run.best_bet.label} ({run.best_bet.bet})
                          </p>
                        )}
                      </article>
                    ))}
                  </div>
                );
              });
            })()}
          </div>
        </section>

        <section className="panel">
          <div className="panel__header">
            <h2>Détails run</h2>
          </div>
          {!selectedRun && <p>Sélectionnez un run pour voir les détails.</p>}
          {selectedRun && (
            <div>
              <h3>{selectedRun.run_id}</h3>
              <p>Créé le {formatDate(selectedRun.created_at)}</p>
              <p>Modèle: {selectedRun.model_label}</p>
              {selectedRun && recommendedList.length > 0 && (
                <div className="alert alert--info">
                  {recommendedList.map((bet, idx) => (
                    <div key={`${bet.matchIndex}-${bet.pivot}-${idx}`}>
                      {bet.label ?? `Pivot ${bet.pivot}`} ({bet.side}) @ {bet.odds ?? "—"} · Edge {(bet.edge * 100).toFixed(2)}% · Prob{" "}
                      {typeof bet.probability === "number" ? (bet.probability * 100).toFixed(1) : "—"}% · Kelly{" "}
                      {typeof bet.kelly === "number" ? (bet.kelly * 100).toFixed(1) : "—"}%
                    </div>
                  ))}
                </div>
              )}
              {selectedRun && recommendedList.length === 0 && (
                <div className="alert alert--error">
                  Aucun pari avec edge positif détecté sur ce run. Garder le bankroll au chaud.
                </div>
              )}
              {selectedRun && safeList.length > 0 && (
                <div className="alert alert--safe">
                  {safeList.map((entry, idx) => {
                    const prob = entry.pick.probability !== undefined ? (entry.pick.probability * 100).toFixed(1) : "—";
                    const edge = entry.pick.edge !== undefined ? (entry.pick.edge * 100).toFixed(2) : "—";
                    const odds = entry.pick.odds !== undefined ? entry.pick.odds.toFixed(2) : "—";
                    const confidence =
                      entry.pick.confidence !== undefined
                        ? entry.pick.confidence
                        : entry.pick.probability !== undefined
                          ? Math.abs(entry.pick.probability - 0.5)
                          : undefined;
                    const confidenceText = confidence !== undefined ? `${(confidence * 100).toFixed(1)}%` : "—";
                    return (
                      <div key={`${entry.pick.pivot}-${entry.pick.side}-${idx}`}>
                       
                        {entry.pick.label ?? `${entry.pick.side} ${entry.pick.pivot}`}
                        {` (${entry.pick.side}) @ ${odds} · Edge ${edge}% · Prob ${prob}% · Confiance ${confidenceText}`}
                      </div>
                    );
                  })}
                </div>
              )}
              {selectedRun && safeList.length === 0 && (
                <div className="alert alert--safe alert--muted">
                  {safeReason || "Aucun pick prudent ne respecte les seuils configurés."}
                </div>
              )}
              <div className="match-list">
                {selectedRun.matches.map((match: MatchEntry, idx: number) => {
                  const best = pickBestEvaluation(match);
                  const highlightPivot = best && Number.isFinite(best.pivot) ? best.pivot : null;
                  const bestEdgeText = typeof best?.edge === "number" ? (best.edge * 100).toFixed(2) : "0.00";
                  const bestSide = best?.side ?? "?";
                  const bestOdds = best?.odds ?? "—";
                  const bestProbability = typeof best?.probability === "number" ? (best.probability * 100).toFixed(1) : "—";
                  const bestKelly = typeof best?.kelly === "number" ? (best.kelly * 100).toFixed(1) : "—";
                  return (
                    <article key={idx} className="match-card">
                    <h4>{match.match ? `${match.match.home?.name} @ ${match.match.away?.name}` : `Match ${idx + 1}`}</h4>
                    <p>Date: {match.match?.match_date}</p>
                    <p>Source: {match.match?.metadata?.source}</p>
                    {match.summary && (
                      <p>
                        Recommendation: {match.summary.recommendation ?? "—"} | Safe edge: {match.summary.safe_edge ?? "—"}
                      </p>
                    )}
                    {highlightPivot !== null && (
                        <p className="match-card__best">
                          Pivot clé {highlightPivot} ({bestSide}) – edge {bestEdgeText}% @ {bestOdds} · Prob {bestProbability}% · Kelly {bestKelly}%
                        </p>
                    )}
                    <details>
                      <summary>Evaluations</summary>
                      <ul>
                        {match.evaluations?.map((evaluation: any, evalIdx: number) => (
                          <li key={evalIdx}>
                            <strong>{evaluation.label}</strong> – pivot {evaluation.pivot}
                            <br />
                            Over edge: {evaluation.over?.edge?.toFixed?.(3)} | Under edge: {evaluation.under?.edge?.toFixed?.(3)}
                          </li>
                        ))}
                      </ul>
                    </details>
                    <PivotChart prediction={match.prediction} highlightPivot={highlightPivot} />
                    <PivotOddsChart evaluations={match.evaluations ?? []} highlightPivot={highlightPivot} />
                    </article>
                  );
                })}
              </div>
            </div>
          )}
          {loadingRun && <p>Chargement...</p>}
        </section>
      </main>
    </div>
  );
};

export default App;
