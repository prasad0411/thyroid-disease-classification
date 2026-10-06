import { useEffect, useState } from 'react';
import { explain, getModelInfo, predict, type Explanation, type ModelInfo, type Prediction } from './api';
import { HistoryPanel } from './components/HistoryPanel';
import { PatientForm } from './components/PatientForm';
import { Report } from './components/Report';
import { useHistory } from './state/useHistory';
import type { PatientInput } from './validation';

type PredictState =
  | { status: 'idle' }
  | { status: 'loading' }
  | {
      status: 'success';
      patientId: string;
      patient: PatientInput;
      reportedAt: string;
      result: Prediction;
      explanation: Explanation | null;
      explainError: string | null;
    }
  | { status: 'error'; message: string };

const messageOf = (err: unknown) => (err instanceof Error ? err.message : 'Request failed');
const stamp = () =>
  new Date().toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' });

export default function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [pred, setPred] = useState<PredictState>({ status: 'idle' });
  const [slow, setSlow] = useState(false);
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

  async function handlePredict(patient: PatientInput, patientId: string) {
    setPred({ status: 'loading' });
    setSlow(false);
    const timer = setTimeout(() => setSlow(true), 2500);
    const [p, e] = await Promise.allSettled([predict(patient), explain(patient)]);
    clearTimeout(timer);
    if (p.status === 'rejected') {
      setPred({ status: 'error', message: messageOf(p.reason) });
      return;
    }
    const reportedAt = stamp();
    setPred({
      status: 'success',
      patientId,
      patient,
      reportedAt,
      result: p.value,
      explanation: e.status === 'fulfilled' ? e.value : null,
      explainError: e.status === 'rejected' ? messageOf(e.reason) : null,
    });
    dispatch({ type: 'add', entry: { at: new Date().toLocaleTimeString(), patientId, patient, result: p.value } });
  }

  return (
    <div className="app">
      <header className="appbar">
        <div className="appbar-inner">
          <div className="brand">
            <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
              <rect width="32" height="32" rx="7" />
              <path d="M16 7c-2.2 3.4-6.5 4.6-6.5 9.6A5.5 5.5 0 0 0 16 22a5.5 5.5 0 0 0 6.5-5.4C22.5 11.6 18.2 10.4 16 7Z" />
              <path d="M16 22v4" />
            </svg>
            <div>
              <p className="brand-name">Thyroid Panel Review</p>
              <p className="brand-dept">Endocrinology decision support</p>
            </div>
          </div>
        </div>
      </header>

      <div className="page">
        {infoError && (
          <div className="service-error" role="alert">
            {infoError}
          </div>
        )}
        <main className="layout">
          <PatientForm submitting={pred.status === 'loading'} onSubmit={handlePredict} />
          <div className="report-slot">
            {pred.status === 'success' && (
              <Report
                patientId={pred.patientId}
                patient={pred.patient}
                reportedAt={pred.reportedAt}
                result={pred.result}
                explanation={pred.explanation}
                explainError={pred.explainError}
                info={info}
              />
            )}
            {pred.status === 'error' && (
              <div className="service-error" role="alert">
                {pred.message}
              </div>
            )}
            {pred.status === 'idle' && (
              <div className="report report-empty">
                <h2 className="empty-title">No panel reviewed yet</h2>
                <p>Enter the thyroid function results on the left, or load an example patient, then select Predict.</p>
              </div>
            )}
            {pred.status === 'loading' && (
              <div className="report report-loading" aria-busy="true">
                <div className="skeleton skeleton-title" />
                <div className="skeleton skeleton-line" />
                <div className="skeleton skeleton-line short" />
                <p className="loading-note">
                  {slow ? 'Starting the model service. The first review after a quiet period can take a few seconds.' : 'Reviewing panel'}
                </p>
              </div>
            )}
          </div>
        </main>
        <HistoryPanel />
      </div>
    </div>
  );
}
