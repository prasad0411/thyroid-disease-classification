import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { HistoryProvider } from './state/HistoryProvider';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HistoryProvider>
      <App />
    </HistoryProvider>
  </StrictMode>,
);
