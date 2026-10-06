import { useEffect, useState } from 'react';
import { explain, getModelInfo, predict, type Explanation, type ModelInfo, type Prediction } from './api';
import { HistoryPanel } from './components/HistoryPanel';
import { ModelInfoBar } from './components/ModelInfoBar';
import { PatientForm } from './components/PatientForm';
import { Report } from './components/Report';
import { useHistory } from './state/useHistory';
import type { PatientInput } from './validation';

type PredictState =
  | { status: 'idle' }
  | { status: 'loading' }
  | {
      status: 'success';
      patient: PatientInput;
      result: Prediction;
      explanation: Explanation | null;
      explainError: string | null;
    }
  | { status: 'error'; message: string };

const messageOf = (err: unknown) => (err instanceof Error ? err.message : 'Request failed');

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
        if (!ctrl.signal.aborted) setInfoError(messageOf(err));
      });
    return () => ctrl.abort();
  }, []);

  async function handlePredict(patient: PatientInput) {
    setPred({ status: 'loading' });
    const [p, e] = await Promise.allSettled([predict(patient), explain(patient)]);
    if (p.status === 'rejected') {
      setPred({ status: 'error', message: messageOf(p.reason) });
      return;
    }
    setPred({
      status: 'success',
      patient,
      result: p.value,
      explanation: e.status === 'fulfilled' ? e.value : null,
      explainError: e.status === 'rejected' ? messageOf(e.reason) : null,
    });
    dispatch({ type: 'add', entry: { at: new Date().toLocaleTimeString(), patient, result: p.value } });
  }

  return (
    <div className="page">
      <header className="masthead">
        <h1>Thyroid panel review</h1>
        <p className="lede">Enter a thyroid function panel to see which pattern the model finds, and why.</p>
      </header>
      {infoError && <ModelInfoBar info={null} error={infoError} />}
      <main className="layout">
        <PatientForm submitting={pred.status === 'loading'} onSubmit={handlePredict} />
        <div className="report-slot">
          {pred.status === 'success' && (
            <Report
              patient={pred.patient}
              result={pred.result}
              explanation={pred.explanation}
              explainError={pred.explainError}
            />
          )}
          {pred.status === 'error' && (
            <div className="service-error" role="alert">
              {pred.message}
            </div>
          )}
          {pred.status === 'idle' && (
            <div className="report report-empty">
              <p>Choose an example patient or enter lab values, then select Predict to generate a report.</p>
            </div>
          )}
          {pred.status === 'loading' && (
            <div className="report report-empty" aria-busy="true">
              <p>Running the model</p>
            </div>
          )}
          {!infoError && <ModelInfoBar info={info} error={null} />}
        </div>
      </main>
      <HistoryPanel />
    </div>
  );
}
