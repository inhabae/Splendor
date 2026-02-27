import { ColorCountsDTO, TokenCountsDTO } from '../../types';

type TokenKey = keyof TokenCountsDTO;

const TOKEN_LABELS: Record<TokenKey, string> = {
  white: 'W',
  blue: 'B',
  green: 'G',
  red: 'R',
  black: 'K',
  gold: 'Gd',
};

export function TokenPill({ color, count }: { color: TokenKey; count: number }) {
  return (
    <div className={`token-pill token-${color}`} aria-label={`${color} token count ${count}`}>
      <span className="token-pill-label">{TOKEN_LABELS[color]}</span>
      <span className="token-pill-count">{count}</span>
    </div>
  );
}

export function ColorBadge({ color, count }: { color: keyof ColorCountsDTO; count: number }) {
  const label = color === 'black' ? 'K' : color[0].toUpperCase();
  return (
    <div className={`color-badge token-${color}`} aria-label={`${color} bonus count ${count}`}>
      <span>{label}</span>
      <strong>{count}</strong>
    </div>
  );
}
