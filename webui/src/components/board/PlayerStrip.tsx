import { PlayerBoardDTO, Seat, TokenCountsDTO } from '../../types';
import { CardView } from './CardView';
import { ColorBadge, TokenPill } from './TokenPill';

const TOKEN_ORDER: Array<keyof TokenCountsDTO> = ['white', 'blue', 'green', 'red', 'black', 'gold'];

export function PlayerStrip({ player, seat }: { player: PlayerBoardDTO; seat: Seat }) {
  return (
    <section className="player-strip" aria-label={`Player ${seat} state`}>
      <div className="player-strip-header">
        <h3>{player.display_name}</h3>
        <div className="point-badge">{player.points}★</div>
        {player.is_to_move && <div className="turn-badge">To Move</div>}
      </div>

      <div className="player-row compact">
        <div className="token-row">
          {TOKEN_ORDER.map((color) => (
            <TokenPill key={`${seat}-tk-${color}`} color={color} count={player.tokens[color]} />
          ))}
        </div>

        <div className="token-row bonus-row">
          <ColorBadge color="white" count={player.bonuses.white} />
          <ColorBadge color="blue" count={player.bonuses.blue} />
          <ColorBadge color="green" count={player.bonuses.green} />
          <ColorBadge color="red" count={player.bonuses.red} />
          <ColorBadge color="black" count={player.bonuses.black} />
        </div>
      </div>

      <div>
        <h4>Reserved ({player.reserved_public.length}/{player.reserved_total})</h4>
        <div className="reserved-row">
          {player.reserved_public.length === 0 && <div className="empty-note">No public reserved cards</div>}
          {player.reserved_public.map((card, idx) => (
            <CardView key={`${seat}-reserved-${idx}-${card.points}-${card.bonus_color}`} card={card} />
          ))}
        </div>
      </div>
    </section>
  );
}
