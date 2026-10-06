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
const EXPLANATION = {
  ...PREDICTION,
  base_value: 0.1,
  method: 'TreeSHAP',
  contributions: [
    { feature: 'TSH', value: 15, shap: 2.4 },
    { feature: 'T4', value: 60, shap: 1.1 },
    { feature: 'age', value: 45, shap: -0.2 },
  ],
};

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

type Routes = Record<string, () => Response | Promise<Response>>;
function routeFetch(routes: Routes) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input);
    const key = Object.keys(routes).find((k) => url.endsWith(k));
    if (!key) throw new TypeError(`unrouted ${url}`);
    return routes[key]();
  });
}

const ALL_OK: Routes = {
  '/model-info': () => json(200, MODEL_INFO),
  '/predict': () => json(200, PREDICTION),
  '/explain': () => json(200, EXPLANATION),
};

function renderApp() {
  return render(
    <HistoryProvider>
      <App />
    </HistoryProvider>,
  );
}

describe('App', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('renders a full report with patient context, flags, drivers, and model details', async () => {
    const fetchMock = routeFetch(ALL_OK);
    const user = userEvent.setup();
    renderApp();
    await user.click(screen.getByRole('button', { name: 'Hypothyroid pattern' }));
    await user.click(screen.getByRole('button', { name: 'Predict' }));

    const report = await screen.findByRole('status');
    expect(within(report).getByRole('heading', { name: 'Pattern consistent with hypothyroidism' })).toBeInTheDocument();
    expect(report).toHaveTextContent('MRN 100517');
    expect(report).toHaveTextContent('45 y, female');
    expect(report).toHaveTextContent('98% model probability');
    expect(report).toHaveTextContent('2 of 5 analytes outside reference intervals');
    const tshRow = within(report).getByRole('row', { name: /^TSH/ });
    expect(tshRow).toHaveTextContent('15.0');
    expect(tshRow).toHaveTextContent('High');
    expect(within(report).getByText('TSH 15')).toBeInTheDocument();
    expect(within(report).getByTestId('model-info')).toHaveTextContent('150,000 synthetic patient records');

    const predictCall = fetchMock.mock.calls.find(([u]) => String(u).endsWith('/predict'));
    expect(JSON.parse(String(predictCall?.[1]?.body))).toMatchObject({ TSH: 15, T4: 60, sex: 0 });
  });

  it('copies a structured note to the clipboard', async () => {
    routeFetch(ALL_OK);
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue(undefined);
    renderApp();
    await user.click(screen.getByRole('button', { name: 'Hypothyroid pattern' }));
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    await user.click(await screen.findByRole('button', { name: 'Copy to note' }));
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0][0]).toContain('TSH: 15.0 mIU/L H (ref 0.4 to 4)');
    expect(await screen.findByRole('button', { name: 'Copied to clipboard' })).toBeInTheDocument();
  });

  it('switches to a plain language patient view', async () => {
    routeFetch(ALL_OK);
    const user = userEvent.setup();
    renderApp();
    await user.click(screen.getByRole('button', { name: 'Hypothyroid pattern' }));
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    await user.click(await screen.findByRole('button', { name: 'Patient' }));
    const report = screen.getByRole('status');
    expect(within(report).getByRole('heading', { name: 'What these results mean' })).toBeInTheDocument();
    expect(report).toHaveTextContent('is higher than the usual range');
    expect(report).toHaveTextContent('influenced most by the TSH level');
    expect(within(report).queryByText('What drove this result')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Clinician' }));
    expect(within(report).getByText('What drove this result')).toBeInTheDocument();
  });

  it('flags an out of range value while typing', async () => {
    routeFetch({ '/model-info': () => json(200, MODEL_INFO) });
    const user = userEvent.setup();
    renderApp();
    const tsh = screen.getByLabelText('TSH');
    await user.clear(tsh);
    await user.type(tsh, '12');
    expect(screen.getAllByText('High').length).toBeGreaterThan(0);
  });

  it('still shows the prediction when the explanation fails', async () => {
    routeFetch({ ...ALL_OK, '/explain': () => json(500, { detail: 'SHAP failed' }) });
    const user = userEvent.setup();
    renderApp();
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    const report = await screen.findByRole('status');
    expect(report).toHaveTextContent('Pattern consistent with hypothyroidism');
    expect(report).toHaveTextContent('Explanation unavailable: SHAP failed');
  });

  it('blocks submission when a lab value is invalid', async () => {
    const fetchMock = routeFetch({ '/model-info': () => json(200, MODEL_INFO) });
    const user = userEvent.setup();
    renderApp();
    const tsh = screen.getByLabelText('TSH');
    await user.clear(tsh);
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    expect(await screen.findByText('Required')).toBeInTheDocument();
    expect(tsh).toHaveAttribute('aria-invalid', 'true');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('shows how to start the API when it is down', async () => {
    routeFetch({ '/model-info': () => new Response('Bad Gateway', { status: 502 }) });
    renderApp();
    expect(await screen.findByRole('alert')).toHaveTextContent('uvicorn api.predict:app --port 8000');
  });
});
