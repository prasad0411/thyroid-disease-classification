import { createContext } from 'react';
import type { HistoryAction, HistoryState } from './historyReducer';

export interface HistoryContextValue {
  state: HistoryState;
  dispatch: (action: HistoryAction) => void;
}

export const HistoryContext = createContext<HistoryContextValue | null>(null);
