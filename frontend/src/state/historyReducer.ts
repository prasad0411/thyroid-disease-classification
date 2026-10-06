import type { Prediction } from '../api';
import type { PatientInput } from '../validation';

export const HISTORY_LIMIT = 20;

export interface HistoryEntry {
  id: number;
  at: string;
  patientId: string;
  patient: PatientInput;
  result: Prediction;
}

export type HistoryAction =
  | { type: 'add'; entry: Omit<HistoryEntry, 'id'> }
  | { type: 'clear' };

export interface HistoryState {
  nextId: number;
  entries: HistoryEntry[];
}

export const initialHistory: HistoryState = { nextId: 1, entries: [] };

export function historyReducer(state: HistoryState, action: HistoryAction): HistoryState {
  switch (action.type) {
    case 'add':
      return {
        nextId: state.nextId + 1,
        entries: [{ ...action.entry, id: state.nextId }, ...state.entries].slice(0, HISTORY_LIMIT),
      };
    case 'clear':
      return { ...state, entries: [] };
  }
}
