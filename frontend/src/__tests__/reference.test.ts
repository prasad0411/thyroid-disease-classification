import { describe, expect, it } from 'vitest';
import { ANALYTES, featureLabel, flagOf, fti, position } from '../reference';

const tsh = ANALYTES.find((a) => a.key === 'TSH')!;
const t4 = ANALYTES.find((a) => a.key === 'T4')!;

describe('reference intervals', () => {
  it('flags high, low, and in range values', () => {
    expect(flagOf(tsh, 15)).toBe('H');
    expect(flagOf(tsh, 0.05)).toBe('L');
    expect(flagOf(tsh, 2.5)).toBeNull();
    expect(flagOf(t4, 120)).toBeNull();
  });

  it('places TSH on a log scale and clamps off scale values', () => {
    expect(position(tsh, 1).at).toBeCloseTo(0.5, 5);
    expect(position(tsh, 500)).toEqual({ at: 1, offScale: true });
    expect(position(t4, 125).at).toBeCloseTo(0.5, 5);
  });

  it('computes FTI with the training formula', () => {
    expect(fti({ T4: 101, T4U: 0.99 })).toBeCloseTo(101, 5);
  });

  it('labels features in plain language', () => {
    expect(featureLabel('TSH', 15)).toBe('TSH 15');
    expect(featureLabel('FTI', 59.40594)).toBe('Free T4 index 59.41');
    expect(featureLabel('sex', 1)).toBe('Sex: male');
    expect(featureLabel('on_thyroxine', 0)).toBe('On thyroxine: no');
  });
});
