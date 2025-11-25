export interface BestBet {
  edge: number;
  label?: string;
  pivot?: number;
  bet?: string;
  probability?: number;
  odds?: number;
  matchup?: string | null;
  kelly?: Record<string, number>;
}

export interface RunSummary {
  run_id: string;
  created_at?: string;
  matches_count: number;
  model_label?: string;
  path?: string;
  best_bet?: BestBet | null;
}

export interface EvaluationKellyDetail {
  full?: number;
  scaled?: Record<string, number>;
  [key: string]: number | Record<string, number> | undefined;
}

export interface MatchEvaluation {
  label?: string;
  pivot?: number;
  over?: { edge?: number; probability?: number; odds?: number; kelly?: EvaluationKellyDetail };
  under?: { edge?: number; probability?: number; odds?: number; kelly?: EvaluationKellyDetail };
}

export interface MatchPrediction {
  pivot_probabilities?: Record<string, number>;
}

export interface RecommendedSummaryEntry {
  pivot?: number;
  label?: string;
  bookmaker?: string | null;
  side?: "over" | "under";
  edge?: number;
  probability?: number;
  kelly?: number;
  odds?: number;
}

export interface SafeSummaryEntry extends RecommendedSummaryEntry {
  confidence?: number;
}

export interface MatchSummary {
  recommended?: RecommendedSummaryEntry[];
  safe?: {
    pick?: SafeSummaryEntry | null;
    reason?: string;
    [key: string]: any;
  };
  recommendation?: string;
  safe_edge?: string;
  [key: string]: unknown;
}

export interface MatchEntry {
  match?: {
    home?: { name?: string };
    away?: { name?: string };
    match_date?: string;
    metadata?: { source?: string };
    markets?: {
      pivot?: number;
      label?: string;
      over_odds?: number;
      under_odds?: number;
    }[];
  };
  evaluations?: MatchEvaluation[];
  prediction?: MatchPrediction;
  summary?: MatchSummary;
}

export interface RunDetail extends RunSummary {
  matches: MatchEntry[];
}

export interface WatcherState {
  state: string;
  heartbeat?: string;
  processed_total?: number;
  last_run_id?: string;
  last_output_path?: string;
  last_processed_at?: string;
  last_source?: string;
  model_label?: string;
  last_error?: {
    message?: string;
    at?: string;
    source?: string;
  } | null;
  pid?: number;
  started_at?: string;
}
