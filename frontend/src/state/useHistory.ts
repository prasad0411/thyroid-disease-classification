import { useContext } from 'react';
import { HistoryContext, type HistoryContextValue } from './historyContext';

export function useHistory(): HistoryContextValue {
  const ctx = useContext(HistoryContext);
  if (!ctx) throw new Error('useHistory must be used inside <HistoryProvider>');
  return ctx;
}
