import type { Contribution, Prediction } from './api';
import { ANALYTES, flagOf, valueOf } from './reference';
import type { PatientInput } from './validation';

const WHAT_IT_IS: Record<string, string> = {
  TSH: 'TSH, the signal that tells the thyroid how hard to work,',
  T3: 'T3, one of the active thyroid hormones,',
  T4: 'Total T4, the main hormone the thyroid makes,',
  T4U: 'T4 uptake, a measure of the proteins that carry thyroid hormone in the blood,',
  FTI: 'The free T4 index, an estimate of hormone available to the body,',
};

const OVERALL: Record<string, string> = {
  hypothyroid: 'Taken together, these results match a pattern often seen when the thyroid is less active than usual (hypothyroidism).',
  hyperthyroid: 'Taken together, these results match a pattern often seen when the thyroid is more active than usual (hyperthyroidism).',
  negative: 'Taken together, these results do not match a pattern of an underactive or overactive thyroid.',
};

const INFLUENCE_NAMES: Record<string, string> = {
  TSH: 'TSH level',
  T3: 'T3 level',
  T4: 'total T4 level',
  T4U: 'T4 uptake',
  FTI: 'free T4 index',
  age: 'age',
  sex: 'sex',
  on_thyroxine: 'thyroxine treatment',
  on_antithyroid: 'antithyroid treatment',
  sick: 'current illness',
  query_hypothyroid: 'clinical suspicion of an underactive thyroid',
  query_hyperthyroid: 'clinical suspicion of an overactive thyroid',
};

export interface PatientSummary {
  findings: string[];
  overall: string;
  influence: string | null;
}

/** Plain language summary a clinician can read with the patient. No treatment advice. */
export function patientSummary(patient: PatientInput, result: Prediction, contributions: Contribution[] | null): PatientSummary {
  const findings = ANALYTES.flatMap((a) => {
    const flag = flagOf(a, valueOf(a, patient));
    if (!flag) return [];
    return [`${WHAT_IT_IS[a.key]} is ${flag === 'H' ? 'higher' : 'lower'} than the usual range.`];
  });
  if (findings.length === 0) findings.push('All measured values are within the usual range.');

  const top = contributions?.find((c) => c.shap > 0);
  const influence = top ? `The result was influenced most by the ${INFLUENCE_NAMES[top.feature] ?? top.feature}.` : null;

  return { findings, overall: OVERALL[result.prediction] ?? '', influence };
}
