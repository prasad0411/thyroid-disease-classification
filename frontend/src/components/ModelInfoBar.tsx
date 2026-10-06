import type { ModelInfo } from '../api';
import { pct } from '../format';

interface Props {
  info: ModelInfo | null;
  error: string | null;
}

export function ModelInfoBar({ info, error }: Props) {
  if (error) {
    return (
      <div className="service-error" role="alert">
        {error}
      </div>
    );
  }
  if (!info) return <p className="model-note" aria-busy="true">Loading model details</p>;
  const m = info.test_metrics;
  return (
    <p className="model-note" data-testid="model-info">
      {info.model} model, version {info.version}. Test accuracy {pct(m.accuracy)} and macro F1 {pct(m.f1_macro)},
      against a {pct(info.majority_baseline)} majority baseline. Trained on{' '}
      {info.dataset_size.toLocaleString('en-US')} synthetic records. Not for clinical use.
    </p>
  );
}
