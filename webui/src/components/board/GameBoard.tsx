import { ActionVizDTO, BoardStateDTO, TokenCountsDTO } from '../../types';
import { NobleView } from './NobleView';
import { PlayerStrip } from './PlayerStrip';
import { TierRow } from './TierRow';
import { TokenPill } from './TokenPill';

const TOKEN_ORDER: Array<keyof TokenCountsDTO> = ['white', 'blue', 'green', 'red', 'black', 'gold'];

export function GameBoard({ board, overlays = [] }: { board: BoardStateDTO; overlays?: ActionVizDTO[] }) {
  return (
    <section className="board-surface">
      <header className="board-meta">
        <div>Target: {board.meta.target_points}</div>
        <div>Turn: {board.meta.turn_index}</div>
        <div>To Move: {board.meta.player_to_move}</div>
      </header>
      <section className="board-main">
        <aside className="board-left">
          <PlayerStrip player={board.players[0]} seat="P0" overlays={overlays} />
          <PlayerStrip player={board.players[1]} seat="P1" overlays={overlays} />
        </aside>

        <section className="board-center">
          <div className="bank-row">
            {TOKEN_ORDER.filter((c) => c !== 'gold').map((color) => (
              <TokenPill
                key={`bank-${color}`}
                color={color}
                count={board.bank[color]}
                overlays={overlays.filter((a) => a.placement_hint.zone === 'bank_token' && a.placement_hint.color === color)}
              />
            ))}
            <TokenPill key="bank-gold" color="gold" count={board.bank.gold} overlays={[]} />
          </div>
          <div className="deck-stack-col">
            {board.tiers.map((tier) => (
              <div key={`deck-${tier.tier}`} className="deck-badge">
                <span>Deck</span>
                <b>{tier.deck_count < 0 ? '?' : tier.deck_count}</b>
              </div>
            ))}
          </div>
        </section>

        <section className="board-right">
          <div className="nobles-row">
            <div className="nobles-grid">
              {board.nobles.length === 0 && <div className="empty-note">No nobles available</div>}
              {board.nobles.map((noble, idx) => (
                <NobleView key={`noble-${idx}`} noble={noble} />
              ))}
            </div>
          </div>
          <div className="tiers-wrap">
            {board.tiers.map((tier) => (
              <TierRow key={`tier-row-${tier.tier}`} tier={tier} overlays={overlays} />
            ))}
          </div>
        </section>
      </section>
    </section>
  );
}
