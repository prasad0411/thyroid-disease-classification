import type { Contribution, Prediction } from './api';
import { pct } from './format';
import { ANALYTES, PATTERN_TEXT, featureLabel, flagOf, valueOf } from './reference';
import type { PatientInput } from './validation';

export function formatResult(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}

/** Plain text summary a clinician can paste into a progress note. */
export function buildNote(opts: {
  patientId: string;
  patient: PatientInput;
  result: Prediction;
  contributions: Contribution[] | null;
  reportedAt: string;
}): string {
  const { patientId, patient, result, contributions, reportedAt } = opts;
  const lines = [
    'Thyroid panel review',
    `Patient: ${patientId || 'not recorded'}, ${patient.age} y, ${patient.sex === 1 ? 'male' : 'female'}`,
    `Reported: ${reportedAt}`,
    '',
    ...ANALYTES.map((a) => {
      const v = valueOf(a, patient);
      const flag = flagOf(a, v);
      return `${a.name}: ${formatResult(v)}${flag ? ` ${flag}` : ''} (ref ${a.low} to ${a.high})`;
    }),
    '',
    `Interpretation: ${PATTERN_TEXT[result.prediction] ?? result.prediction} (model probability ${pct(result.confidence, 0)}).`,
  ];
  if (contributions?.length) {
    const top = contributions.slice(0, 3).map((c) => `${featureLabel(c.feature, c.value)} (${c.shap >= 0 ? '+' : '\u2212'}${Math.abs(c.shap).toFixed(2)})`);
    lines.push(`Main contributors: ${top.join(', ')}.`);
  }
  return lines.join('\n');
}
