import { TierRowDTO } from '../../types';
import { CardView } from './CardView';

export function TierRow({ tier }: { tier: TierRowDTO }) {
  return (
    <section className="tier-row" aria-label={`Tier ${tier.tier} cards`}>
      <div className="tier-header"><h4>Tier {tier.tier}</h4></div>
      <div className="tier-cards">
        {tier.cards.length === 0 && <div className="empty-note">No visible cards</div>}
        {tier.cards.map((card, idx) => (
          <CardView key={`tier-${tier.tier}-${idx}-${card.points}-${card.bonus_color}`} card={card} />
        ))}
      </div>
    </section>
  );
}
