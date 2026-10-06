import type { Prediction } from '../api';
import { pct } from '../format';

export function ResultCard({ result }: { result: Prediction }) {
  const rows = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <section className={`card result result-${result.prediction}`} role="status" aria-live="polite">
      <p className="eyebrow">Predicted class</p>
      <h2 className="prediction">{result.prediction}</h2>
      <p className="confidence">Confidence {pct(result.confidence)}</p>
      <ul className="bars" aria-label="Class probabilities">
        {rows.map(([cls, p]) => (
          <li key={cls}>
            <span className="bar-label">{cls}</span>
            <span className="bar-track">
              <span className={`bar-fill fill-${cls}`} style={{ width: `${Math.max(p * 100, 0.5)}%` }} />
            </span>
            <span className="bar-value">{pct(p)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
