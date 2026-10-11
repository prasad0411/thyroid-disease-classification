import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ApolloProvider } from '@apollo/client';
import App from './App';
import { makeClient } from './graphql';
import { HistoryProvider } from './state/HistoryProvider';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ApolloProvider client={makeClient()}>
      <HistoryProvider>
        <App />
      </HistoryProvider>
    </ApolloProvider>
  </StrictMode>,
);
