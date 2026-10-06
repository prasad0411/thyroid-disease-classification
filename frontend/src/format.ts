export function pct(x: number | null | undefined, digits = 1): string {
  return typeof x === 'number' && Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : 'n/a';
}
