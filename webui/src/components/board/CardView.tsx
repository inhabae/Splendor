import { CardDTO } from '../../types';

const COST_ORDER: Array<keyof CardDTO['cost']> = ['white', 'blue', 'green', 'red', 'black'];

export function CardView({ card }: { card: CardDTO }) {
  const summary = `Card ${card.bonus_color} bonus, ${card.points} points`;
  const reqs = COST_ORDER.filter((color) => card.cost[color] > 0);
  const bonusLabel =
    card.bonus_color === 'black' ? '+' : card.bonus_color === 'white' ? '▼' : card.bonus_color[0].toUpperCase();
  return (
    <article className={`card-view card-${card.bonus_color}`} aria-label={summary}>
      <header className="card-head">
        <span className="card-points">{card.points}</span>
        <span className="card-bonus">{bonusLabel}</span>
      </header>
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
