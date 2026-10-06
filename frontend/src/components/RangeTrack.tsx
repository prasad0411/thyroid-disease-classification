import { position, type Analyte, type Flag } from '../reference';

interface Props {
  analyte: Analyte;
  value: number;
  flag: Flag;
}

export function RangeTrack({ analyte, value, flag }: Props) {
  const lo = position(analyte, analyte.low).at;
  const hi = position(analyte, analyte.high).at;
  const { at, offScale } = position(analyte, value);
  const label = `${analyte.name} ${value.toFixed(2)}, reference ${analyte.low} to ${analyte.high}${
    flag === 'H' ? ', high' : flag === 'L' ? ', low' : ', within range'
  }${analyte.log ? ', log scale' : ''}`;
  return (
    <div className="track" role="img" aria-label={label}>
      <span className="track-band" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
      <span
        className={`track-dot${flag ? ` dot-${flag}` : ''}${offScale ? ' dot-off' : ''}`}
        style={{ left: `${at * 100}%` }}
      />
    </div>
  );
}
