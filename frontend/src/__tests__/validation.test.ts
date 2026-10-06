import { describe, expect, it } from 'vitest';
import { DEFAULT_PATIENT, labsToStrings, validateLabs } from '../validation';

describe('validateLabs', () => {
  it('accepts the default patient', () => {
    const { values, errors } = validateLabs(labsToStrings(DEFAULT_PATIENT));
    expect(errors).toEqual({});
    expect(values?.TSH).toBe(2.5);
  });

  it('flags empty, non numeric, and out of range values', () => {
    const raw = { ...labsToStrings(DEFAULT_PATIENT), TSH: '', T3: 'abc', age: '130' };
    const { values, errors } = validateLabs(raw);
    expect(values).toBeNull();
    expect(errors).toEqual({ TSH: 'Required', T3: 'Enter a number', age: 'Must be between 0 and 120' });
  });

  it('accepts range boundaries', () => {
    const raw = { ...labsToStrings(DEFAULT_PATIENT), TSH: '0', T4: '500' };
    expect(validateLabs(raw).errors).toEqual({});
  });
});
