import { describe, expect, it } from 'vitest';
import { buildNote, formatResult } from '../note';
import { DEFAULT_PATIENT } from '../validation';

const result = {
  prediction: 'hypothyroid',
  confidence: 0.98,
  probabilities: { hypothyroid: 0.98, negative: 0.02, hyperthyroid: 0 },
  features_used: [],
  model_version: 'v',
};

describe('buildNote', () => {
  it('writes flagged results, interpretation, and top contributors', () => {
    const note = buildNote({
      patientId: 'MRN 100517',
      patient: { ...DEFAULT_PATIENT, TSH: 15, T4: 60 },
      result,
      contributions: [
        { feature: 'TSH', value: 15, shap: 2.29 },
        { feature: 'T4', value: 60, shap: 1.69 },
        { feature: 'T3', value: 1.8, shap: -0.48 },
        { feature: 'age', value: 45, shap: 0.03 },
      ],
      reportedAt: 'Oct 6, 2026, 1:20 PM',
    });
    expect(note).toContain('Patient: MRN 100517, 45 y, female');
    expect(note).toContain('TSH: 15.0 H (ref 0.4 to 4)');
    expect(note).toContain('Total T4: 60.0 (ref 60 to 120)');
    expect(note).toContain('Free T4 index: 59.4 L (ref 60 to 120)');
    expect(note).toContain('Interpretation: Pattern consistent with hypothyroidism (model probability 98%).');
    expect(note).toContain('Main contributors: TSH 15 (+2.29), Total T4 60 (+1.69), T3 1.8 (\u22120.48).');
    expect(note).not.toContain('Age 45');
  });

  it('handles a missing ID and no explanation', () => {
    const note = buildNote({ patientId: '', patient: DEFAULT_PATIENT, result, contributions: null, reportedAt: 'now' });
    expect(note).toContain('Patient: not recorded');
    expect(note).not.toContain('Main contributors');
  });

  it('formats results by magnitude', () => {
    expect(formatResult(190)).toBe('190');
    expect(formatResult(15)).toBe('15.0');
    expect(formatResult(0.05)).toBe('0.05');
  });
});
