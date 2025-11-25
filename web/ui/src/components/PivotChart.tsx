import type { MatchPrediction } from "../types";

interface PivotChartProps {
  prediction?: MatchPrediction;
  highlightPivot?: number | null;
}

const width = 480;
const height = 220;
const padding = 48;

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

export const PivotChart = ({ prediction, highlightPivot }: PivotChartProps) => {
  const pivotProbabilities = prediction?.pivot_probabilities;
  if (!pivotProbabilities) {
    return null;
  }

  const points = Object.entries(pivotProbabilities)
    .map(([pivot, probability]) => ({
      pivot: Number(pivot),
      probability: Number(probability),
    }))
    .filter((point) => Number.isFinite(point.pivot) && Number.isFinite(point.probability))
    .sort((a, b) => a.pivot - b.pivot);

  if (!points.length) {
    return null;
  }

  const pivots = points.map((p) => p.pivot);
  const probabilities = points.map((p) => p.probability);
  const minPivot = Math.min(...pivots);
  const maxPivot = Math.max(...pivots);
  const minProb = Math.min(...probabilities);
  const maxProb = Math.max(...probabilities);
  const pivotRange = maxPivot - minPivot || 1;
  const probRange = maxProb - minProb || 1;

  const scaleX = (pivot: number) => padding + ((pivot - minPivot) / pivotRange) * (width - padding * 2);
  const scaleY = (probability: number) =>
    height - padding - ((probability - minProb) / probRange) * (height - padding * 1.5);

  const pathD = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${scaleX(point.pivot)},${scaleY(point.probability)}`)
    .join(" ");

  const yAxisTicks = [minProb, (minProb + maxProb) / 2, maxProb];
  const xTickIndices = [0, Math.floor(points.length / 2), points.length - 1];
  const xTicks = Array.from(new Set(xTickIndices.map((idx) => points[idx]?.pivot))).filter(
    (value): value is number => typeof value === "number",
  );

  const highlightX =
    typeof highlightPivot === "number" && highlightPivot >= minPivot && highlightPivot <= maxPivot
      ? scaleX(highlightPivot)
      : null;

  return (
    <div className="pivot-chart">
      <div className="pivot-chart__header">
        <strong>Courbe des probabilités cumulées</strong>
        <span>
          {minPivot} → {maxPivot}
        </span>
      </div>
      <svg width={width} height={height} role="img" aria-label="pivot probabilities">
        <rect
          x={padding}
          y={padding / 2}
          width={width - padding * 2}
          height={height - padding * 1.5}
          className="pivot-chart__plot"
        />
        {highlightX && (
          <line
            x1={highlightX}
            x2={highlightX}
            y1={padding / 2}
            y2={height - padding / 2}
            className="pivot-chart__highlight"
          />
        )}
        <path d={pathD} className="pivot-chart__line" fill="none" />
        {points.map((point, idx) => (
          <circle
            key={`${point.pivot}-${idx}`}
            cx={scaleX(point.pivot)}
            cy={scaleY(point.probability)}
            r={3}
            className="pivot-chart__point"
          />
        ))}
        {yAxisTicks.map((tick, idx) => (
          <g key={idx}>
            <line x1={padding - 6} x2={padding} y1={scaleY(tick)} y2={scaleY(tick)} className="pivot-chart__axis" />
            <text x={padding - 10} y={scaleY(tick) + 4} textAnchor="end" className="pivot-chart__label">
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
              className="pivot-chart__axis"
            />
            <text x={scaleX(tick)} y={height - padding + 20} textAnchor="middle" className="pivot-chart__label">
              {tick}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};
