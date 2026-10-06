import { useState, type FormEvent } from 'react';
import {
  DEFAULT_PATIENT,
  FLAG_FIELDS,
  LAB_FIELDS,
  SEX_OPTIONS,
  labsToStrings,
  validateLabs,
  type Flag,
  type FlagField,
  type LabErrors,
  type LabStrings,
  type PatientInput,
} from '../validation';

const EXAMPLES: { label: string; patient: PatientInput }[] = [
  { label: 'Typical normal', patient: DEFAULT_PATIENT },
  { label: 'Hypothyroid pattern', patient: { ...DEFAULT_PATIENT, TSH: 15, T4: 60 } },
  {
    label: 'Hyperthyroid pattern',
    patient: { ...DEFAULT_PATIENT, age: 35, sex: 1, TSH: 0.05, T3: 4.5, T4: 190, query_hyperthyroid: 1 },
  },
];

type Flags = Record<FlagField, Flag>;

function flagsOf(p: PatientInput): Flags {
  const { sex, on_thyroxine, on_antithyroid, sick, query_hypothyroid, query_hyperthyroid } = p;
  return { sex, on_thyroxine, on_antithyroid, sick, query_hypothyroid, query_hyperthyroid };
}

interface Props {
  submitting: boolean;
  onSubmit: (patient: PatientInput) => void;
}

export function PatientForm({ submitting, onSubmit }: Props) {
  const [labs, setLabs] = useState<LabStrings>(labsToStrings(DEFAULT_PATIENT));
  const [flags, setFlags] = useState<Flags>(flagsOf(DEFAULT_PATIENT));
  const [errors, setErrors] = useState<LabErrors>({});

  function load(p: PatientInput) {
    setLabs(labsToStrings(p));
    setFlags(flagsOf(p));
    setErrors({});
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const { values, errors: found } = validateLabs(labs);
    setErrors(found);
    if (values) onSubmit({ ...values, ...flags });
  }

  return (
    <form className="entry" onSubmit={handleSubmit} noValidate aria-label="Patient lab panel">
      <h2 className="entry-title">Patient panel</h2>
      <div className="examples" role="group" aria-label="Load an example patient">
        <span className="examples-label">Examples</span>
        {EXAMPLES.map((ex) => (
          <button key={ex.label} type="button" className="chip" onClick={() => load(ex.patient)}>
            {ex.label}
          </button>
        ))}
      </div>

      <fieldset>
        <legend>Lab values</legend>
        <div className="grid">
          {LAB_FIELDS.map((spec) => {
            const err = errors[spec.key];
            const id = `lab-${spec.key}`;
            return (
              <div key={spec.key} className="field">
                <label htmlFor={id}>{spec.label}</label>
                <input
                  id={id}
                  type="number"
                  inputMode="decimal"
                  step={spec.step}
                  min={spec.min}
                  max={spec.max}
                  value={labs[spec.key]}
                  aria-invalid={err ? true : undefined}
                  aria-describedby={err ? `${id}-err` : undefined}
                  onChange={(e) => setLabs((prev) => ({ ...prev, [spec.key]: e.target.value }))}
                />
                {err && (
                  <span id={`${id}-err`} className="field-error">
                    {err}
                  </span>
                )}
              </div>
            );
          })}
          <div className="field">
            <label htmlFor="sex">Sex</label>
            <select
              id="sex"
              value={flags.sex}
              onChange={(e) => setFlags((prev) => ({ ...prev, sex: Number(e.target.value) as Flag }))}
            >
              {SEX_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Clinical history</legend>
        <div className="flags">
          {FLAG_FIELDS.map((f) => (
            <label key={f.key} className="check">
              <input
                type="checkbox"
                checked={flags[f.key] === 1}
                onChange={(e) => setFlags((prev) => ({ ...prev, [f.key]: e.target.checked ? 1 : 0 }))}
              />
              {f.label}
            </label>
          ))}
        </div>
      </fieldset>

      <div className="actions">
        <button type="submit" className="primary" disabled={submitting}>
          {submitting ? 'Predicting…' : 'Predict'}
        </button>
        <button type="button" onClick={() => load(DEFAULT_PATIENT)} disabled={submitting}>
          Reset
        </button>
      </div>
    </form>
  );
}
