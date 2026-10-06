import type { PatientInput } from './validation';

export type Flag = 'H' | 'L' | null;

export interface Analyte {
  key: 'TSH' | 'T3' | 'T4' | 'T4U' | 'FTI';
  name: string;
  low: number;
  high: number;
  /** Display domain for the interval track. */
  min: number;
  max: number;
  log?: boolean;
  derived?: boolean;
}

// Intervals mirror REF_RANGES in app.py so both apps flag values identically.
export const ANALYTES: readonly Analyte[] = [
  { key: 'TSH', name: 'TSH', low: 0.4, high: 4.0, min: 0.01, max: 100, log: true },
  { key: 'T3', name: 'T3', low: 0.8, high: 2.0, min: 0, max: 6 },
  { key: 'T4', name: 'Total T4', low: 60, high: 120, min: 0, max: 250 },
  { key: 'T4U', name: 'T4 uptake', low: 0.7, high: 1.2, min: 0, max: 2.5 },
  { key: 'FTI', name: 'Free T4 index', low: 60, high: 120, min: 0, max: 250, derived: true },
];

export function fti(p: Pick<PatientInput, 'T4' | 'T4U'>): number {
  return p.T4 / (p.T4U + 0.01); // same formula as data_generator.py
}

export function valueOf(a: Analyte, p: PatientInput): number {
  return a.key === 'FTI' ? fti(p) : p[a.key];
}

export function flagOf(a: Analyte, v: number): Flag {
  if (v > a.high) return 'H';
  if (v < a.low) return 'L';
  return null;
}

/** Position on the track as a 0..1 fraction, clamped, plus whether it fell off the scale. */
export function position(a: Analyte, v: number): { at: number; offScale: boolean } {
  const f = (x: number) => (a.log ? Math.log10(Math.max(x, a.min)) : x);
  const raw = (f(v) - f(a.min)) / (f(a.max) - f(a.min));
  return { at: Math.min(1, Math.max(0, raw)), offScale: raw < 0 || raw > 1 };
}

const FEATURE_NAMES: Record<string, string> = {
  TSH: 'TSH',
  T3: 'T3',
  T4: 'Total T4',
  T4U: 'T4 uptake',
  FTI: 'Free T4 index',
  age: 'Age',
  sex: 'Sex',
  on_thyroxine: 'On thyroxine',
  on_antithyroid: 'On antithyroid medication',
  sick: 'Currently unwell',
  query_hypothyroid: 'Clinician suspects hypothyroid',
  query_hyperthyroid: 'Clinician suspects hyperthyroid',
};

const BINARY = new Set(['on_thyroxine', 'on_antithyroid', 'sick', 'query_hypothyroid', 'query_hyperthyroid']);

export function featureLabel(feature: string, value: number): string {
  const name = FEATURE_NAMES[feature] ?? feature;
  if (feature === 'sex') return `Sex: ${value === 1 ? 'male' : 'female'}`;
  if (BINARY.has(feature)) return `${name}: ${value === 1 ? 'yes' : 'no'}`;
  const shown = Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, '');
  return `${name} ${shown}`;
}

export const PATTERN_TEXT: Record<string, string> = {
  hypothyroid: 'Pattern consistent with hypothyroidism',
  hyperthyroid: 'Pattern consistent with hyperthyroidism',
  negative: 'No thyroid dysfunction pattern',
};
