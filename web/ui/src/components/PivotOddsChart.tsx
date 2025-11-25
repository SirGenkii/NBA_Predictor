import type { MatchEvaluation } from "../types";

interface PivotOddsChartProps {
  evaluations: MatchEvaluation[];
  highlightPivot?: number | null;
}

const width = 480;
const height = 230;
const padding = 48;

const toImplied = (odds?: number) => {
  if (!odds || odds <= 0) return null;
  return 1 / odds;
};

const formatPercent = (value?: number | null) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
};

export const PivotOddsChart = ({ evaluations, highlightPivot }: PivotOddsChartProps) => {
  if (!evaluations?.length) return null;

  const points = evaluations
    .filter((evaluation) => typeof evaluation.pivot === "number")
    .map((evaluation) => ({
      pivot: Number(evaluation.pivot),
      over_model: typeof evaluation.over?.probability === "number" ? evaluation.over.probability : null,
      under_model: typeof evaluation.under?.probability === "number" ? evaluation.under.probability : null,
      over_implied: toImplied(evaluation.over?.odds),
      under_implied: toImplied(evaluation.under?.odds),
      label: evaluation.label,
    }))
    .sort((a, b) => a.pivot - b.pivot);

  if (!points.length) {
    return null;
  }

  const minPivot = points[0].pivot;
  const maxPivot = points[points.length - 1].pivot;
  const pivotRange = maxPivot - minPivot || 1;

  const scaleX = (pivot: number) => padding + ((pivot - minPivot) / pivotRange) * (width - padding * 2);
  const scaleY = (value: number) => height - padding - value * (height - padding * 1.5);

  const pathFor = (key: "over_model" | "under_model" | "over_implied" | "under_implied") => {
    const entries = points.filter((point) => typeof point[key] === "number");
    if (!entries.length) return null;
    return entries
      .map((point, index) => `${index === 0 ? "M" : "L"}${scaleX(point.pivot)},${scaleY(point[key] as number)}`)
      .join(" ");
  };

  const overModelPath = pathFor("over_model");
  const underModelPath = pathFor("under_model");
  const overImpliedPath = pathFor("over_implied");
  const underImpliedPath = pathFor("under_implied");

  const xTickIndices = [0, Math.floor(points.length / 2), points.length - 1];
  const xTicks = Array.from(new Set(xTickIndices.map((idx) => points[idx]?.pivot))).filter(
    (value): value is number => typeof value === "number",
  );

  const highlightX =
    typeof highlightPivot === "number" && highlightPivot >= minPivot && highlightPivot <= maxPivot
      ? scaleX(highlightPivot)
      : null;

  return (
    <div className="pivot-odds">
      <div className="pivot-odds__header">
        <strong>Probabilité modèle vs implied odds</strong>
        <span>
          {minPivot} → {maxPivot}
        </span>
      </div>
      <svg width={width} height={height} role="img" aria-label="comparaison modèle et cotes">
        <rect
          x={padding}
          y={padding / 2}
          width={width - padding * 2}
          height={height - padding * 1.5}
          className="pivot-odds__plot"
        />
        {highlightX && (
          <line
            x1={highlightX}
            x2={highlightX}
            y1={padding / 2}
            y2={height - padding / 2}
            className="pivot-odds__highlight"
          />
        )}
        {overModelPath && <path d={overModelPath} className="pivot-odds__line pivot-odds__line--over" fill="none" />}
        {underModelPath && <path d={underModelPath} className="pivot-odds__line pivot-odds__line--under" fill="none" />}
        {overImpliedPath && (
          <path
            d={overImpliedPath}
            className="pivot-odds__line pivot-odds__line--over implied"
            fill="none"
            strokeDasharray="4 4"
          />
        )}
        {underImpliedPath && (
          <path
            d={underImpliedPath}
            className="pivot-odds__line pivot-odds__line--under implied"
            fill="none"
            strokeDasharray="4 4"
          />
        )}
        {points.map((point, idx) => (
          <g key={idx}>
            {typeof point.over_model === "number" && (
              <circle
                cx={scaleX(point.pivot)}
                cy={scaleY(point.over_model)}
                r={3}
                className="pivot-odds__point pivot-odds__point--over"
              />
            )}
            {typeof point.under_model === "number" && (
              <circle
                cx={scaleX(point.pivot)}
                cy={scaleY(point.under_model)}
                r={3}
                className="pivot-odds__point pivot-odds__point--under"
              />
            )}
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={padding - 6} x2={padding} y1={scaleY(tick)} y2={scaleY(tick)} className="pivot-odds__axis" />
            <text x={padding - 10} y={scaleY(tick) + 4} textAnchor="end" className="pivot-odds__label">
              {formatPercent(tick)}
            </text>
          </g>
        ))}
        {xTicks.map((tick, idx) => (
          <g key={`x-${idx}`}>
            <line
              x1={scaleX(tick)}
              x2={scaleX(tick)}
              y1={height - padding}
              y2={height - padding + 6}
              className="pivot-odds__axis"
            />
            <text x={scaleX(tick)} y={height - padding + 20} textAnchor="middle" className="pivot-odds__label">
              {tick}
            </text>
          </g>
        ))}
      </svg>
      <div className="pivot-odds__legend">
        <span>
          <i className="pivot-odds__dot pivot-odds__dot--over" /> Over modèle
        </span>
        <span>
          <i className="pivot-odds__dot pivot-odds__dot--under" /> Under modèle
        </span>
        <span>
          <i className="pivot-odds__dot pivot-odds__dot--over implied" /> Over implied
        </span>
        <span>
          <i className="pivot-odds__dot pivot-odds__dot--under implied" /> Under implied
        </span>
      </div>
    </div>
  );
};
