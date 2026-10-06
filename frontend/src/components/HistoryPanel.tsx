import { pct } from '../format';
import { PATTERN_TEXT } from '../reference';
import { useHistory } from '../state/useHistory';

export function HistoryPanel() {
  const { state, dispatch } = useHistory();
  if (state.entries.length === 0) return null;
  return (
    <section className="history" aria-labelledby="history-heading">
      <div className="history-head">
        <h3 id="history-heading" className="section-title">Reviewed this session</h3>
        <button type="button" onClick={() => dispatch({ type: 'clear' })}>
          Clear
        </button>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Patient</th>
              <th className="num">TSH</th>
              <th className="num">T3</th>
              <th className="num">T4</th>
              <th>Interpretation</th>
              <th className="num">Probability</th>
            </tr>
          </thead>
          <tbody>
            {state.entries.map((e) => (
              <tr key={e.id}>
                <td>{e.at}</td>
                <td>{e.patientId || 'Not recorded'}</td>
                <td className="num">{e.patient.TSH}</td>
                <td className="num">{e.patient.T3}</td>
                <td className="num">{e.patient.T4}</td>
                <td>{PATTERN_TEXT[e.result.prediction] ?? e.result.prediction}</td>
                <td className="num">{pct(e.result.confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
