import type { PatientInput } from './validation';

export interface ModelInfo {
  model: string;
  version: string;
  dataset_size: number;
  majority_baseline: number | null;
  test_metrics: { accuracy?: number; f1_macro?: number; log_loss?: number };
  features: string[];
}

export interface Prediction {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  features_used: string[];
  model_version: string;
}

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const BASE: string = import.meta.env?.VITE_API_BASE ?? '/api';
const TIMEOUT_MS = 10_000;

interface FastApiValidationItem {
  loc?: (string | number)[];
  msg?: string;
}

export function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: FastApiValidationItem) => {
        const field = (d.loc ?? []).filter((p) => p !== 'body').join('.');
        return field ? `${field}: ${d.msg ?? 'invalid'}` : d.msg ?? 'invalid';
      })
      .join('; ');
  }
  return 'Request failed';
}

async function request<T>(path: string, init: RequestInit, fetchImpl: typeof fetch, signal?: AbortSignal): Promise<T> {
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), TIMEOUT_MS);
  const onAbort = () => timeout.abort();
  signal?.addEventListener('abort', onAbort);
  let res: Response;
  try {
    res = await fetchImpl(`${BASE}${path}`, { ...init, signal: timeout.signal });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new ApiError(0, 'Cannot reach the prediction service. Is the API running on port 8000?');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === 'object' && 'detail' in body) message = formatDetail((body as { detail: unknown }).detail);
    } catch {
      /* non JSON error body: keep the status message */
    }
    throw new ApiError(res.status, message);
  }
  return (await res.json()) as T;
}

export function getModelInfo(signal?: AbortSignal, fetchImpl: typeof fetch = fetch): Promise<ModelInfo> {
  return request<ModelInfo>('/model-info', { method: 'GET' }, fetchImpl, signal);
}

export function predict(patient: PatientInput, fetchImpl: typeof fetch = fetch): Promise<Prediction> {
  return request<Prediction>(
    '/predict',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patient) },
    fetchImpl,
  );
}
