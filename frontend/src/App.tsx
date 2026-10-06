import { useEffect, useState } from 'react';
import { ApiError, getModelInfo, predict, type ModelInfo, type Prediction } from './api';
import { HistoryPanel } from './components/HistoryPanel';
import { ModelInfoBar } from './components/ModelInfoBar';
import { PatientForm } from './components/PatientForm';
import { ResultCard } from './components/ResultCard';
import { useHistory } from './state/useHistory';
import type { PatientInput } from './validation';

type PredictState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; result: Prediction }
  | { status: 'error'; message: string };

export default function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [pred, setPred] = useState<PredictState>({ status: 'idle' });
  const { dispatch } = useHistory();

  useEffect(() => {
    const ctrl = new AbortController();
    getModelInfo(ctrl.signal)
      .then(setInfo)
      .catch((err: unknown) => {
        if (!ctrl.signal.aborted) setInfoError(err instanceof Error ? err.message : 'Unknown error');
      });
    return () => ctrl.abort();
  }, []);

  async function handlePredict(patient: PatientInput) {
    setPred({ status: 'loading' });
    try {
      const result = await predict(patient);
      setPred({ status: 'success', result });
      dispatch({ type: 'add', entry: { at: new Date().toLocaleTimeString(), patient, result } });
    } catch (err) {
      const message = err instanceof ApiError || err instanceof Error ? err.message : 'Prediction failed';
      setPred({ status: 'error', message });
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <h1>Thyroid Classification</h1>
        <p className="subtitle">Enter a lab panel to classify hypothyroid, hyperthyroid, or negative.</p>
      </header>
      <ModelInfoBar info={info} error={infoError} />
      <main className="layout">
        <PatientForm submitting={pred.status === 'loading'} onSubmit={handlePredict} />
        <div className="side">
          {pred.status === 'success' && <ResultCard result={pred.result} />}
          {pred.status === 'error' && (
            <div className="card banner-error" role="alert">
              {pred.message}
            </div>
          )}
          {pred.status === 'idle' && <div className="card muted">Results appear here.</div>}
          {pred.status === 'loading' && (
            <div className="card muted" aria-busy="true">
              Running model…
            </div>
          )}
        </div>
      </main>
      <HistoryPanel />
    </div>
  );
}
