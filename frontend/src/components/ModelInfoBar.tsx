import type { ModelInfo } from '../api';
import { pct } from '../format';

interface Props {
  info: ModelInfo | null;
  error: string | null;
}

export function ModelInfoBar({ info, error }: Props) {
  if (error) {
    return (
      <div className="banner banner-error" role="alert">
        Model service unavailable: {error}
      </div>
    );
  }
  if (!info) return <div className="banner" aria-busy="true">Loading model details…</div>;
  const m = info.test_metrics;
  return (
    <div className="banner" data-testid="model-info">
      <dl className="stats">
        <div><dt>Model</dt><dd>{info.model}</dd></div>
        <div><dt>Test accuracy</dt><dd>{pct(m.accuracy)}</dd></div>
        <div><dt>Macro F1</dt><dd>{pct(m.f1_macro)}</dd></div>
        <div><dt>Majority baseline</dt><dd>{pct(info.majority_baseline)}</dd></div>
        <div><dt>Training data</dt><dd>{info.dataset_size.toLocaleString('en-US')} synthetic records</dd></div>
      </dl>
      <p className="disclaimer">Research demo trained on synthetic data. Not for clinical use. Model version {info.version}.</p>
    </div>
  );
}
