import { useState, type FormEvent } from 'react';
import { ANALYTES, flagOf } from '../reference';
import { displayUnit } from '../note';
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
  type LabField,
  type LabStrings,
  type PatientInput,
} from '../validation';

const EXAMPLES: { label: string; id: string; patient: PatientInput }[] = [
  { label: 'Typical normal', id: 'MRN 100482', patient: DEFAULT_PATIENT },
  { label: 'Hypothyroid pattern', id: 'MRN 100517', patient: { ...DEFAULT_PATIENT, TSH: 15, T4: 60 } },
  {
    label: 'Hyperthyroid pattern',
    id: 'MRN 100563',
    patient: { ...DEFAULT_PATIENT, age: 35, sex: 1, TSH: 0.05, T3: 4.5, T4: 190, query_hyperthyroid: 1 },
  },
];

type Flags = Record<FlagField, Flag>;

function flagsOf(p: PatientInput): Flags {
  const { sex, on_thyroxine, on_antithyroid, sick, query_hypothyroid, query_hyperthyroid } = p;
  return { sex, on_thyroxine, on_antithyroid, sick, query_hypothyroid, query_hyperthyroid };
}

const SPEC = Object.fromEntries(LAB_FIELDS.map((s) => [s.key, s])) as Record<LabField, (typeof LAB_FIELDS)[number]>;
const REF = Object.fromEntries(ANALYTES.map((a) => [a.key, a])) as Record<string, (typeof ANALYTES)[number]>;
const PANEL: LabField[] = ['TSH', 'T3', 'T4', 'T4U'];

interface Props {
  submitting: boolean;
  onSubmit: (patient: PatientInput, patientId: string) => void;
}

export function PatientForm({ submitting, onSubmit }: Props) {
  const [patientId, setPatientId] = useState('');
  const [labs, setLabs] = useState<LabStrings>(labsToStrings(DEFAULT_PATIENT));
  const [flags, setFlags] = useState<Flags>(flagsOf(DEFAULT_PATIENT));
  const [errors, setErrors] = useState<LabErrors>({});

  function load(p: PatientInput, id: string) {
    setPatientId(id);
    setLabs(labsToStrings(p));
    setFlags(flagsOf(p));
    setErrors({});
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const { values, errors: found } = validateLabs(labs);
    setErrors(found);
    if (values) onSubmit({ ...values, ...flags }, patientId.trim());
  }

  function labInput(key: LabField, hint?: string) {
    const spec = SPEC[key];
    const err = errors[key];
    const id = `lab-${key}`;
    const describedBy = [err ? `${id}-err` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ') || undefined;
    const ref = REF[key];
    const n = Number(labs[key]);
    const liveFlag = ref && labs[key].trim() !== '' && Number.isFinite(n) ? flagOf(ref, n) : null;
    const unit = ref ? displayUnit(ref.unit) : '';
    return (
      <div key={key} className="field">
        <div className="label-row">
          <label htmlFor={id}>{spec.label}</label>
          {liveFlag && <span className={`live-flag pill-${liveFlag}`}>{liveFlag === 'H' ? 'High' : 'Low'}</span>}
        </div>
        <div className={`input-group${unit ? ' has-suffix' : ''}`}>
        <input
          id={id}
          type="number"
          inputMode="decimal"
          step={spec.step}
          min={spec.min}
          max={spec.max}
          value={labs[key]}
          aria-invalid={err ? true : undefined}
          aria-describedby={describedBy}
          onChange={(e) => setLabs((prev) => ({ ...prev, [key]: e.target.value }))}
        />
        {unit && <span className="input-suffix" aria-hidden="true">{unit}</span>}
        </div>
        {err ? (
          <span id={`${id}-err`} className="field-error">{err}</span>
        ) : (
          hint && <span id={`${id}-hint`} className="field-hint">{hint}</span>
        )}
      </div>
    );
  }

  return (
    <form className="order" onSubmit={handleSubmit} noValidate aria-label="Patient lab panel">
      <div className="order-head">
        <h2 className="order-title">Thyroid function panel</h2>
        <div className="examples" role="group" aria-label="Load an example patient">
          <span className="examples-label">Load example</span>
          {EXAMPLES.map((ex) => (
            <button key={ex.label} type="button" className="chip" onClick={() => load(ex.patient, ex.id)}>
              {ex.label}
            </button>
          ))}
        </div>
      </div>

      <fieldset>
        <legend>Patient</legend>
        <div className="grid grid-patient">
          <div className="field">
            <label htmlFor="patient-id">Patient ID</label>
            <input
              id="patient-id"
              type="text"
              maxLength={32}
              autoComplete="off"
              placeholder="MRN or reference"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
            />
          </div>
          {labInput('age')}
          <div className="field">
            <label htmlFor="sex">Sex</label>
            <select
              id="sex"
              value={flags.sex}
              onChange={(e) => setFlags((prev) => ({ ...prev, sex: Number(e.target.value) as Flag }))}
            >
              {SEX_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Lab values</legend>
        <div className="grid grid-labs">
          {PANEL.map((k) => labInput(k, `Ref ${REF[k].low} to ${REF[k].high}${displayUnit(REF[k].unit) ? ` ${displayUnit(REF[k].unit)}` : ''}`))}
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
        <button type="button" onClick={() => load(DEFAULT_PATIENT, '')} disabled={submitting}>
          Reset
        </button>
      </div>
    </form>
  );
}
