import { useMemo, useReducer, type ReactNode } from 'react';
import { HistoryContext } from './historyContext';
import { historyReducer, initialHistory } from './historyReducer';

export function HistoryProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(historyReducer, initialHistory);
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <HistoryContext.Provider value={value}>{children}</HistoryContext.Provider>;
}
