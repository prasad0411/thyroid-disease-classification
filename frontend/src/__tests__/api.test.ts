import { describe, expect, it, vi } from 'vitest';
import { ApiError, formatDetail, predict } from '../api';
import { DEFAULT_PATIENT } from '../validation';

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('api client', () => {
  it('posts the patient and returns the prediction', async () => {
    const result = { prediction: 'negative', confidence: 0.9, probabilities: { negative: 0.9 }, features_used: [], model_version: 'v1' };
    const fetchMock = vi.fn().mockResolvedValue(json(200, result));
    await expect(predict(DEFAULT_PATIENT, fetchMock)).resolves.toEqual(result);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/predict');
    expect(JSON.parse(init.body)).toEqual(DEFAULT_PATIENT);
  });

  it('turns FastAPI 422 details into a readable message', async () => {
    const detail = [{ loc: ['body', 'sex'], msg: 'Input should be 0 or 1' }];
    const fetchMock = vi.fn().mockResolvedValue(json(422, { detail }));
    await expect(predict(DEFAULT_PATIENT, fetchMock)).rejects.toMatchObject({ status: 422, message: 'sex: Input should be 0 or 1' });
  });

  it('reports an unreachable service as status 0', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    const err = await predict(DEFAULT_PATIENT, fetchMock).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
  });

  it('formats string and unknown details', () => {
    expect(formatDetail('missing model features')).toBe('missing model features');
    expect(formatDetail(42)).toBe('Request failed');
  });
});
