import { describe, expect, it } from 'vitest';
import { patientSummary } from '../patient';
import { DEFAULT_PATIENT } from '../validation';

const result = (prediction: string) => ({
  prediction,
  confidence: 0.9,
  probabilities: {},
  features_used: [],
  model_version: 'v',
});

describe('patientSummary', () => {
  it('explains abnormal values and the pattern in plain language', () => {
    const s = patientSummary({ ...DEFAULT_PATIENT, TSH: 15, T4: 60 }, result('hypothyroid'), [
      { feature: 'TSH', value: 15, shap: 2.3 },
      { feature: 'T3', value: 1.8, shap: -0.5 },
    ]);
    expect(s.findings).toHaveLength(2);
    expect(s.findings[0]).toContain('TSH, the signal that tells the thyroid how hard to work, is higher');
    expect(s.findings[1]).toContain('free T4 index');
    expect(s.overall).toContain('less active than usual (hypothyroidism)');
    expect(s.influence).toBe('The result was influenced most by the TSH level.');
  });

  it('reassures when everything is in range and skips negative drivers', () => {
    const s = patientSummary(DEFAULT_PATIENT, result('negative'), [{ feature: 'TSH', value: 2.5, shap: -1 }]);
    expect(s.findings).toEqual(['All measured values are within the usual range.']);
    expect(s.overall).toContain('do not match');
    expect(s.influence).toBeNull();
  });
});
