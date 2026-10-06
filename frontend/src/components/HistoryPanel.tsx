import { pct } from '../format';
import { useHistory } from '../state/useHistory';

export function HistoryPanel() {
  const { state, dispatch } = useHistory();
  if (state.entries.length === 0) return null;
  return (
    <section className="card">
      <div className="history-head">
        <h3>Session history</h3>
        <button type="button" onClick={() => dispatch({ type: 'clear' })}>
          Clear
        </button>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>TSH</th>
              <th>T3</th>
              <th>T4</th>
              <th>Prediction</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {state.entries.map((e) => (
              <tr key={e.id}>
                <td>{e.at}</td>
                <td>{e.patient.TSH}</td>
                <td>{e.patient.T3}</td>
                <td>{e.patient.T4}</td>
                <td>{e.result.prediction}</td>
                <td>{pct(e.result.confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
