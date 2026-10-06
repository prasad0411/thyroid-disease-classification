import type { Contribution } from '../api';
import { featureLabel } from '../reference';

interface Props {
  contributions: Contribution[] | null;
  error: string | null;
  className: string;
  limit?: number;
}

export function Explanation({ contributions, error, className, limit = 6 }: Props) {
  if (error) {
    return <p className="explain-note">Explanation unavailable: {error}</p>;
  }
  if (!contributions) return null;
  const top = contributions.slice(0, limit);
  const scale = Math.max(...top.map((c) => Math.abs(c.shap)), 1e-9);
  return (
    <section aria-labelledby="explain-heading">
      <h3 id="explain-heading">What drove this result</h3>
      <p className="explain-note">
        Bars to the right moved the model toward {className}; bars to the left moved it away.
      </p>
      <ul className="drivers">
        {top.map((c) => {
          const toward = c.shap >= 0;
          const width = `${(Math.abs(c.shap) / scale) * 50}%`;
          return (
            <li key={c.feature} className={toward ? 'toward' : 'away'}>
              <span className="driver-label">{featureLabel(c.feature, c.value)}</span>
              <span className="driver-axis" aria-hidden="true">
                <span className="driver-bar" style={toward ? { left: '50%', width } : { right: '50%', width }} />
              </span>
              <span className="driver-value">
                {toward ? '+' : '\u2212'}
                {Math.abs(c.shap).toFixed(2)}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
