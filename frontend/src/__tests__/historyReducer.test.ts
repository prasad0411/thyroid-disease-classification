import { describe, expect, it } from 'vitest';
import { HISTORY_LIMIT, historyReducer, initialHistory } from '../state/historyReducer';
import { DEFAULT_PATIENT } from '../validation';

const entry = (n: number) => ({
  at: `t${n}`,
  patientId: 'MRN 1',
  patient: DEFAULT_PATIENT,
  result: { prediction: 'negative', confidence: n / 100, probabilities: {}, features_used: [], model_version: 'v' },
});

describe('historyReducer', () => {
  it('adds newest first and caps the list', () => {
    let s = initialHistory;
    for (let i = 0; i < HISTORY_LIMIT + 5; i++) s = historyReducer(s, { type: 'add', entry: entry(i) });
    expect(s.entries).toHaveLength(HISTORY_LIMIT);
    expect(s.entries[0].at).toBe(`t${HISTORY_LIMIT + 4}`);
    expect(new Set(s.entries.map((e) => e.id)).size).toBe(HISTORY_LIMIT);
  });

  it('clears entries but keeps ids increasing', () => {
    let s = historyReducer(initialHistory, { type: 'add', entry: entry(1) });
    s = historyReducer(s, { type: 'clear' });
    expect(s.entries).toEqual([]);
    s = historyReducer(s, { type: 'add', entry: entry(2) });
    expect(s.entries[0].id).toBe(2);
  });
});
