export type LabField = 'TSH' | 'T3' | 'T4' | 'T4U' | 'age';
export type FlagField =
  | 'sex'
  | 'on_thyroxine'
  | 'on_antithyroid'
  | 'sick'
  | 'query_hypothyroid'
  | 'query_hyperthyroid';
export type Flag = 0 | 1;
export type PatientInput = Record<LabField, number> & Record<FlagField, Flag>;
export type LabStrings = Record<LabField, string>;
export type LabErrors = Partial<Record<LabField, string>>;

export interface LabSpec {
  key: LabField;
  label: string;
  min: number;
  max: number;
  step: number;
}

// Ranges mirror the pydantic Field constraints in api/predict.py.
export const LAB_FIELDS: readonly LabSpec[] = [
  { key: 'TSH', label: 'TSH', min: 0, max: 500, step: 0.01 },
  { key: 'T3', label: 'T3', min: 0, max: 20, step: 0.01 },
  { key: 'T4', label: 'Total T4', min: 0, max: 500, step: 0.1 },
  { key: 'T4U', label: 'T4 uptake (T4U)', min: 0, max: 5, step: 0.01 },
  { key: 'age', label: 'Age', min: 0, max: 120, step: 1 },
];

export const FLAG_FIELDS: readonly { key: Exclude<FlagField, 'sex'>; label: string }[] = [
  { key: 'on_thyroxine', label: 'On thyroxine' },
  { key: 'on_antithyroid', label: 'On antithyroid medication' },
  { key: 'sick', label: 'Currently unwell' },
  { key: 'query_hypothyroid', label: 'Clinician suspects hypothyroid' },
  { key: 'query_hyperthyroid', label: 'Clinician suspects hyperthyroid' },
];

// Encoding must match data_generator.py.
export const SEX_OPTIONS: readonly { value: Flag; label: string }[] = [
  { value: 0, label: 'Female' },
  { value: 1, label: 'Male' },
];

export const DEFAULT_PATIENT: PatientInput = {
  TSH: 2.5,
  T3: 1.8,
  T4: 105,
  T4U: 1.0,
  age: 45,
  sex: 0,
  on_thyroxine: 0,
  on_antithyroid: 0,
  sick: 0,
  query_hypothyroid: 0,
  query_hyperthyroid: 0,
};

export function labsToStrings(p: PatientInput): LabStrings {
  return {
    TSH: String(p.TSH),
    T3: String(p.T3),
    T4: String(p.T4),
    T4U: String(p.T4U),
    age: String(p.age),
  };
}

export function validateLabs(raw: LabStrings): { values: Record<LabField, number> | null; errors: LabErrors } {
  const errors: LabErrors = {};
  const values = {} as Record<LabField, number>;
  for (const spec of LAB_FIELDS) {
    const text = raw[spec.key].trim();
    if (text === '') {
      errors[spec.key] = 'Required';
      continue;
    }
    const n = Number(text);
    if (!Number.isFinite(n)) {
      errors[spec.key] = 'Enter a number';
    } else if (n < spec.min || n > spec.max) {
      errors[spec.key] = `Must be between ${spec.min} and ${spec.max}`;
    } else {
      values[spec.key] = n;
    }
  }
  return { values: Object.keys(errors).length === 0 ? values : null, errors };
}
