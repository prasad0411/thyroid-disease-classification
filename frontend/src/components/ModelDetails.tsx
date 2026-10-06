import type { ModelInfo } from '../api';
import { pct } from '../format';

export function ModelDetails({ info }: { info: ModelInfo | null }) {
  if (!info) return null;
  const m = info.test_metrics;
  return (
    <details className="model-details" data-testid="model-info">
      <summary>About this model</summary>
      <dl>
        <div><dt>Model</dt><dd>{info.model}, version {info.version}</dd></div>
        <div><dt>Held out test accuracy</dt><dd>{pct(m.accuracy)}</dd></div>
        <div><dt>Macro F1</dt><dd>{pct(m.f1_macro)}</dd></div>
        <div><dt>Majority class baseline</dt><dd>{pct(info.majority_baseline)}</dd></div>
        <div><dt>Training data</dt><dd>{info.dataset_size.toLocaleString('en-US')} synthetic patient records</dd></div>
        <div><dt>Explanations</dt><dd>TreeSHAP on the gradient boosted model</dd></div>
      </dl>
    </details>
  );
}
