import { useState } from 'react';
import type { Explanation as ExplanationData, ModelInfo, Prediction } from '../api';
import { pct } from '../format';
import { buildNote, displayUnit, formatResult } from '../note';
import { patientSummary } from '../patient';
import { ANALYTES, PATTERN_TEXT, flagOf, valueOf } from '../reference';
import type { PatientInput } from '../validation';
import { Explanation } from './Explanation';
import { ModelDetails } from './ModelDetails';
import { ProbabilityBar } from './ProbabilityBar';
import { RangeTrack } from './RangeTrack';

interface Props {
  patientId: string;
  patient: PatientInput;
  reportedAt: string;
  result: Prediction;
  explanation: ExplanationData | null;
  explainError: string | null;
  info: ModelInfo | null;
}

export function Report({ patientId, patient, reportedAt, result, explanation, explainError, info }: Props) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [view, setView] = useState<'clinician' | 'patient'>('clinician');
  const summary = patientSummary(patient, result, explanation?.contributions ?? null);
  const abnormal = ANALYTES.filter((a) => flagOf(a, valueOf(a, patient)) !== null).length;

  async function copyNote() {
    const text = buildNote({ patientId, patient, result, contributions: explanation?.contributions ?? null, reportedAt });
    try {
      await navigator.clipboard.writeText(text);
      setCopyState('copied');
    } catch {
      setCopyState('failed');
    }
  }

  return (
    <article className={`report report-${result.prediction}`} role="status" aria-live="polite">
      <div className="print-head" aria-hidden="true">
        <strong>Thyroid Panel Review</strong>
        <span>Endocrinology decision support report</span>
      </div>
      <header className="report-bar">
        <dl className="report-meta">
          <div><dt>Patient</dt><dd>{patientId || 'Not recorded'}</dd></div>
          <div><dt>Age and sex</dt><dd>{patient.age} y, {patient.sex === 1 ? 'male' : 'female'}</dd></div>
          <div><dt>Reported</dt><dd>{reportedAt}</dd></div>
        </dl>
        <div className="report-actions">
          <div className="segmented" role="group" aria-label="Report view">
            <button type="button" aria-pressed={view === 'clinician'} onClick={() => setView('clinician')}>Clinician</button>
            <button type="button" aria-pressed={view === 'patient'} onClick={() => setView('patient')}>Patient</button>
          </div>
          <button type="button" onClick={copyNote}>
            {copyState === 'copied' ? 'Copied to clipboard' : 'Copy to note'}
          </button>
          <button type="button" onClick={() => window.print()}>
            Print
          </button>
        </div>
      </header>
      {copyState === 'failed' && (
        <p className="inline-error" role="alert">Clipboard access was blocked by the browser. Use Print instead.</p>
      )}

      <section className="interpretation" aria-labelledby="interp-heading">
        <p className="interp-kicker">Interpretation</p>
        <h2 id="interp-heading" className="pattern">{PATTERN_TEXT[result.prediction] ?? result.prediction}</h2>
        <p className="probability">
          <span className="probability-value">{pct(result.confidence, 0)}</span> model probability
          <span className="abnormal-count">
            {abnormal === 0 ? 'All analytes within reference intervals' : `${abnormal} of ${ANALYTES.length} analytes outside reference intervals`}
          </span>
        </p>
        <ProbabilityBar probabilities={result.probabilities} />
      </section>

      <section aria-labelledby="results-heading">
        <h3 id="results-heading" className="section-title">Results</h3>
        <div className="table-scroll">
          <table className="analytes">
            <thead>
              <tr>
                <th scope="col">Analyte</th>
                <th scope="col" className="num">Result</th>
                <th scope="col" className="flag-col">Flag</th>
                <th scope="col" className="interval-col">Reference interval</th>
              </tr>
            </thead>
            <tbody>
              {ANALYTES.map((a) => {
                const v = valueOf(a, patient);
                const flag = flagOf(a, v);
                return (
                  <tr key={a.key} className={flag ? 'row-abnormal' : undefined}>
                    <th scope="row">
                      {a.name}
                      {a.derived && <span className="derived">calculated</span>}
                    </th>
                    <td className={`num${flag ? ` flagged-${flag}` : ''}`}>
                      {formatResult(v)}
                      {displayUnit(a.unit) && <span className="unit">{displayUnit(a.unit)}</span>}
                    </td>
                    <td className="flag-col">
                      {flag && <span className={`flag-pill pill-${flag}`}>{flag === 'H' ? 'High' : 'Low'}</span>}
                    </td>
                    <td className="interval-col">
                      <RangeTrack analyte={a} value={v} flag={flag} />
                      <span className="interval-text">{a.low} to {a.high}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {view === 'clinician' ? (
        <Explanation
          contributions={explanation?.contributions ?? null}
          error={explainError}
          className={result.prediction}
        />
      ) : (
        <section className="patient-view" aria-labelledby="patient-heading">
          <h3 id="patient-heading" className="section-title">What these results mean</h3>
          <ul className="patient-findings">
            {summary.findings.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
          <p className="patient-overall">{summary.overall}</p>
          {summary.influence && <p className="patient-influence">{summary.influence}</p>}
          <p className="patient-next">Your clinician will go through these results with you and decide whether any further tests or follow up are needed.</p>
        </section>
      )}

      <footer className="report-foot">
        <p className="cds-note">Decision support output. Interpret alongside the history, examination, and other investigations.</p>
        <ModelDetails info={info} />
      </footer>
    </article>
  );
}
