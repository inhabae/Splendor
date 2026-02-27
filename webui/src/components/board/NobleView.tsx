import { NobleDTO } from '../../types';

const REQ_ORDER: Array<keyof NobleDTO['requirements']> = ['white', 'blue', 'green', 'red', 'black'];

export function NobleView({ noble }: { noble: NobleDTO }) {
  const reqs = REQ_ORDER.filter((color) => noble.requirements[color] > 0);
  return (
    <article className="noble-view" aria-label={`Noble worth ${noble.points} points`}>
      <header className="noble-head">{noble.points}</header>
      <div className="noble-reqs">
        {reqs.map((color) => (
          <span key={color} className={`req-chip cost-circle token-${color}`}>
            <b>{noble.requirements[color]}</b>
            <small>{color === 'black' ? '+' : color === 'white' ? '▼' : color[0].toUpperCase()}</small>
          </span>
        ))}
      </div>
    </article>
  );
}
