import { ActionVizDTO, CardDTO } from '../../types';

const COST_ORDER: Array<keyof CardDTO['cost']> = ['white', 'blue', 'green', 'red', 'black'];

function topOverlays(overlays: ActionVizDTO[]): ActionVizDTO[] {
  return overlays
    .slice()
    .sort((a, b) => b.policy_prob - a.policy_prob)
    .slice(0, 3);
}

export function CardView({ card, overlays = [] }: { card: CardDTO; overlays?: ActionVizDTO[] }) {
  const isPrivate = card.source === 'reserved_private';
  const summary = `Card ${card.bonus_color} bonus, ${card.points} points`;
  const reqs = COST_ORDER.filter((color) => card.cost[color] > 0);
  const bonusLabel =
    card.bonus_color === 'black' ? '+' : card.bonus_color === 'white' ? '▼' : card.bonus_color[0].toUpperCase();
  const hints = topOverlays(overlays);
  return (
    <article className={`card-view card-${card.bonus_color} ${isPrivate ? 'card-private' : ''}`} aria-label={summary}>
      <header className="card-head">
        <span className="card-points">{card.points}</span>
        <span className="card-bonus">{bonusLabel}</span>
      </header>
      {hints.length > 0 && (
        <div className="overlay-badges">
          {hints.map((hint) => (
            <span
              key={`overlay-${hint.action_idx}`}
              className={`overlay-badge ${hint.masked ? 'masked' : ''} ${hint.is_selected ? 'selected' : ''}`}
              title={`${hint.label} (${(hint.policy_prob * 100).toFixed(1)}%)`}
            >
              a{hint.action_idx}:{(hint.policy_prob * 100).toFixed(1)}%
            </span>
          ))}
        </div>
      )}
      <div className="card-costs">
        {reqs.map((color) => (
          <span key={color} className={`cost-chip cost-circle token-${color}`}>
            <b>{card.cost[color]}</b>
            <small>{color === 'black' ? '+' : color === 'white' ? '▼' : color[0].toUpperCase()}</small>
          </span>
        ))}
      </div>
    </article>
  );
}
