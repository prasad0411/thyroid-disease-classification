import type { Explanation as ExplanationData, Prediction } from '../api';
import { pct } from '../format';
import { ANALYTES, PATTERN_TEXT, flagOf, valueOf } from '../reference';
import type { PatientInput } from '../validation';
import { Explanation } from './Explanation';
import { RangeTrack } from './RangeTrack';

interface Props {
  patient: PatientInput;
  result: Prediction;
  explanation: ExplanationData | null;
  explainError: string | null;
}

function fmt(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}

export function Report({ patient, result, explanation, explainError }: Props) {
  const others = Object.entries(result.probabilities)
    .filter(([cls]) => cls !== result.prediction)
    .sort((a, b) => b[1] - a[1]);
  return (
    <section className={`report report-${result.prediction}`} role="status" aria-live="polite">
      <header className="report-head">
        <h2 className="pattern">{PATTERN_TEXT[result.prediction] ?? result.prediction}</h2>
        <p className="probability">
          <span className="probability-value">{pct(result.confidence, 0)}</span> model probability
        </p>
        <p className="alternatives">
          {others.map(([cls, p]) => `${cls} ${pct(p, 0)}`).join(', ')}
        </p>
      </header>

      <table className="analytes">
        <caption className="visually-hidden">Patient values against reference intervals</caption>
        <thead>
          <tr>
            <th scope="col">Analyte</th>
            <th scope="col" className="num">Result</th>
            <th scope="col" className="flag-col"><span className="visually-hidden">Flag</span></th>
            <th scope="col" className="interval-col">Reference interval</th>
          </tr>
        </thead>
        <tbody>
          {ANALYTES.map((a) => {
            const v = valueOf(a, patient);
            const flag = flagOf(a, v);
            return (
              <tr key={a.key}>
                <th scope="row">
                  {a.name}
                  {a.derived && <span className="derived">calculated</span>}
                </th>
                <td className={`num${flag ? ` flagged-${flag}` : ''}`}>{fmt(v)}</td>
                <td className={`flag-col flagged-${flag ?? 'none'}`}>{flag ?? ''}</td>
                <td className="interval-col">
                  <RangeTrack analyte={a} value={v} flag={flag} />
                  <span className="interval-text">
                    {a.low} to {a.high}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <Explanation
        contributions={explanation?.contributions ?? null}
        error={explainError}
        className={result.prediction}
      />
    </section>
  );
}
