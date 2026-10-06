import { pct } from '../format';

const ORDER = ['negative', 'hypothyroid', 'hyperthyroid'];
const NAMES: Record<string, string> = { negative: 'Negative', hypothyroid: 'Hypothyroid', hyperthyroid: 'Hyperthyroid' };

export function ProbabilityBar({ probabilities }: { probabilities: Record<string, number> }) {
  const entries = ORDER.filter((k) => k in probabilities).map((k) => [k, probabilities[k]] as const);
  return (
    <div className="probs">
      <div className="probs-bar" role="img" aria-label={entries.map(([k, p]) => `${NAMES[k]} ${pct(p, 0)}`).join(', ')}>
        {entries.map(([k, p]) => (
          <span key={k} className={`probs-seg seg-${k}`} style={{ width: `${p * 100}%` }} />
        ))}
      </div>
      <ul className="probs-legend">
        {entries.map(([k, p]) => (
          <li key={k}>
            <span className={`swatch seg-${k}`} aria-hidden="true" />
            {NAMES[k]} <strong>{pct(p, 0)}</strong>
          </li>
        ))}
      </ul>
    </div>
  );
}
