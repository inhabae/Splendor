import { ActionVizDTO, ColorCountsDTO, TokenCountsDTO } from '../../types';

type TokenKey = keyof TokenCountsDTO;

const TOKEN_LABELS: Record<TokenKey, string> = {
  white: 'W',
  blue: 'B',
  green: 'G',
  red: 'R',
  black: 'K',
  gold: 'Gd',
};

export function TokenPill({ color, count, overlays = [] }: { color: TokenKey; count: number; overlays?: ActionVizDTO[] }) {
  return (
    <div className={`token-pill token-${color}`} aria-label={`${color} token count ${count}`}>
      <span className="token-pill-label">{TOKEN_LABELS[color]}</span>
      <span className="token-pill-count">{count}</span>
      {overlays.slice(0, 2).map((hint) => (
        <span
          key={`token-overlay-${color}-${hint.action_idx}`}
          className={`overlay-badge ${hint.masked ? 'masked' : ''} ${hint.is_selected ? 'selected' : ''}`}
          title={`${hint.label} (${(hint.policy_prob * 100).toFixed(1)}%)`}
        >
          a{hint.action_idx}
        </span>
      ))}
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
