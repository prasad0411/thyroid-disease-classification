import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';
import { HistoryProvider } from '../state/HistoryProvider';

const MODEL_INFO = {
  model: 'XGBoost',
  version: '20260903_155027',
  dataset_size: 150000,
  majority_baseline: 0.7297,
  test_metrics: { accuracy: 0.8509, f1_macro: 0.7213 },
  features: [],
};
const PREDICTION = {
  prediction: 'hypothyroid',
  confidence: 0.98,
  probabilities: { hyperthyroid: 0, hypothyroid: 0.98, negative: 0.02 },
  features_used: [],
  model_version: '20260903_155027',
};

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

function renderApp() {
  return render(
    <HistoryProvider>
      <App />
    </HistoryProvider>,
  );
}

describe('App', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('shows honest model metrics from the API', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(200, MODEL_INFO));
    renderApp();
    const bar = await screen.findByTestId('model-info');
    expect(bar).toHaveTextContent('85.1%');
    expect(bar).toHaveTextContent('150,000 synthetic records');
    expect(bar).toHaveTextContent('Not for clinical use');
  });

  it('submits a lab panel and records it in history', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(json(200, MODEL_INFO))
      .mockResolvedValueOnce(json(200, PREDICTION));
    const user = userEvent.setup();
    renderApp();
    await user.click(screen.getByRole('button', { name: 'Hypothyroid pattern' }));
    await user.click(screen.getByRole('button', { name: 'Predict' }));

    const status = await screen.findByRole('status');
    expect(within(status).getByRole('heading', { name: 'hypothyroid' })).toBeInTheDocument();
    expect(status).toHaveTextContent('Confidence 98.0%');
    const body = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
    expect(body).toMatchObject({ TSH: 15, T4: 60, sex: 0 });
    expect(screen.getByRole('table')).toHaveTextContent('hypothyroid');
  });

  it('blocks submission when a lab value is invalid', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(200, MODEL_INFO));
    const user = userEvent.setup();
    renderApp();
    const tsh = screen.getByLabelText('TSH');
    await user.clear(tsh);
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    expect(await screen.findByText('Required')).toBeInTheDocument();
    expect(tsh).toHaveAttribute('aria-invalid', 'true');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('shows a clear error when the API is down', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
    renderApp();
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the prediction service');
  });
});
