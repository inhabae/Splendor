import { ActionVizDTO, TierRowDTO } from '../../types';
import { CardView } from './CardView';

export function TierRow({
  tier,
  mctsTopAction,
  modelTopAction,
}: {
  tier: TierRowDTO;
  mctsTopAction?: ActionVizDTO | null;
  modelTopAction?: ActionVizDTO | null;
}) {
  const mctsTier = mctsTopAction?.placement_hint.zone === 'faceup_card' ? mctsTopAction.placement_hint.tier : undefined;
  const mctsSlot = mctsTopAction?.placement_hint.zone === 'faceup_card' ? mctsTopAction.placement_hint.slot : undefined;
  const modelTier = modelTopAction?.placement_hint.zone === 'faceup_card' ? modelTopAction.placement_hint.tier : undefined;
  const modelSlot = modelTopAction?.placement_hint.zone === 'faceup_card' ? modelTopAction.placement_hint.slot : undefined;
  return (
    <section className="tier-row" aria-label={`Tier ${tier.tier} cards`}>
      <div className="tier-header"><h4>Tier {tier.tier}</h4></div>
      <div className="tier-cards">
        {tier.cards.length === 0 && <div className="empty-note">No visible cards</div>}
        {tier.cards.map((card, idx) => (
          <CardView
            key={`tier-${tier.tier}-${idx}-${card.points}-${card.bonus_color}`}
            card={card}
            showMcts={mctsTier === tier.tier && mctsSlot != null && card.slot === mctsSlot}
            showModel={modelTier === tier.tier && modelSlot != null && card.slot === modelSlot}
          />
        ))}
      </div>
    </section>
  );
}
