from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpendeeSelectorConfig:
    board_root: str = "[data-spendee-board]"
    bank_root: str = "[data-spendee-bank]"
    player_roots: dict[str, str] = field(
        default_factory=lambda: {
            "P0": "[data-player-seat='P0']",
            "P1": "[data-player-seat='P1']",
        }
    )
    faceup_row_roots: tuple[str, str, str] = (
        "[data-faceup-tier='1']",
        "[data-faceup-tier='2']",
        "[data-faceup-tier='3']",
    )
    noble_root: str = "[data-visible-nobles]"
    turn_indicator_root: str = "[data-current-turn]"
    modal_root: str = "[data-spendee-modal]"
    animation_selector: str = "[data-animating='true']"
    faceup_card_selector_template: str = "[data-faceup-card][data-tier='{tier}'][data-slot='{slot}']"
    reserved_card_selector_template: str = "[data-player-seat='{seat}'] [data-reserved-slot='{slot}']"
    reserve_deck_selector_template: str = "[data-action='reserve-deck'][data-tier='{tier}']"
    gem_button_selector_template: str = "[data-gem-button][data-color='{color}']"
    pass_button_selector: str = "[data-action='pass']"
    noble_option_selector_template: str = "[data-noble-option='{slot}']"


def build_probe_script(config: SpendeeSelectorConfig) -> str:
    config_json = json.dumps(
        {
            "boardRoot": config.board_root,
            "bankRoot": config.bank_root,
            "playerRoots": config.player_roots,
            "faceupRowRoots": list(config.faceup_row_roots),
            "nobleRoot": config.noble_root,
            "turnIndicatorRoot": config.turn_indicator_root,
            "modalRoot": config.modal_root,
            "animationSelector": config.animation_selector,
        }
    )
    return f"""
(() => {{
  const cfg = {config_json};
  const colors = ["white", "blue", "green", "red", "black", "gold"];

  const readCountMap = (root, attrName) => {{
    const out = {{}};
    if (!root) {{
      return out;
    }}
    for (const color of colors) {{
      const el = root.querySelector(`[data-${{attrName}}='${{color}}']`);
      out[color] = el ? Number(el.getAttribute("data-count") || el.textContent || 0) : 0;
    }}
    return out;
  }};

  const readCard = (node) => {{
    if (!node) {{
      return null;
    }}
    return {{
      tier: Number(node.getAttribute("data-tier") || 0),
      points: Number(node.getAttribute("data-points") || 0),
      bonus_color: String(node.getAttribute("data-bonus-color") || ""),
      cost: {{
        white: Number(node.getAttribute("data-cost-white") || 0),
        blue: Number(node.getAttribute("data-cost-blue") || 0),
        green: Number(node.getAttribute("data-cost-green") || 0),
        red: Number(node.getAttribute("data-cost-red") || 0),
        black: Number(node.getAttribute("data-cost-black") || 0),
      }},
      is_private: String(node.getAttribute("data-private") || "false") === "true",
    }};
  }};

  const readNoble = (node) => {{
    if (!node) {{
      return null;
    }}
    return {{
      points: Number(node.getAttribute("data-points") || 3),
      requirements: {{
        white: Number(node.getAttribute("data-req-white") || 0),
        blue: Number(node.getAttribute("data-req-blue") || 0),
        green: Number(node.getAttribute("data-req-green") || 0),
        red: Number(node.getAttribute("data-req-red") || 0),
        black: Number(node.getAttribute("data-req-black") || 0),
      }},
    }};
  }};

  const readReservedSlots = (root) => {{
    const slots = [];
    if (!root) {{
      return slots;
    }}
    for (let slot = 0; slot < 3; slot += 1) {{
      const slotRoot = root.querySelector(`[data-reserved-slot='${{slot}}']`);
      if (!slotRoot) {{
        slots.push({{ slot, state: "empty" }});
        continue;
      }}
      const hidden = String(slotRoot.getAttribute("data-hidden") || "false") === "true";
      const tierHint = Number(slotRoot.getAttribute("data-tier-hint") || 0);
      const cardNode = slotRoot.querySelector("[data-card]");
      slots.push(
        hidden
          ? {{ slot, state: "hidden", tier_hint: tierHint || null }}
          : {{ slot, state: "visible", card: readCard(cardNode ?? slotRoot) }}
      );
    }}
    return slots;
  }};

  const readPlayer = (seat) => {{
    const root = document.querySelector(cfg.playerRoots[seat]);
    if (!root) {{
      return null;
    }}
    return {{
      seat,
      points: Number(root.getAttribute("data-points") || root.querySelector("[data-points]")?.textContent || 0),
      tokens: readCountMap(root, "token-color"),
      bonuses: readCountMap(root, "bonus-color"),
      purchased_cards: Array.from(root.querySelectorAll("[data-purchased-card]")).map(readCard).filter(Boolean),
      reserved_slots: readReservedSlots(root),
      claimed_nobles: Array.from(root.querySelectorAll("[data-claimed-noble]")).map(readNoble).filter(Boolean),
    }};
  }};

  const faceup = cfg.faceupRowRoots.map((rowSelector) => {{
    const rowRoot = document.querySelector(rowSelector);
    if (!rowRoot) {{
      return {{ deck_count: 0, cards: [null, null, null, null] }};
    }}
    const cards = [0, 1, 2, 3].map((slot) => {{
      const slotRoot = rowRoot.querySelector(`[data-faceup-slot='${{slot}}']`);
      if (!slotRoot) {{
        return null;
      }}
      return readCard(slotRoot.querySelector("[data-card]") ?? slotRoot);
    }});
    return {{
      deck_count: Number(rowRoot.getAttribute("data-deck-count") || rowRoot.querySelector("[data-deck-count]")?.textContent || 0),
      cards,
    }};
  }});

  const modalRoot = document.querySelector(cfg.modalRoot);
  const modal = modalRoot
    ? {{
        kind: String(modalRoot.getAttribute("data-modal-kind") || "none"),
        options: Array.from(modalRoot.querySelectorAll("[data-modal-option]")).map((node) => {{
          const noble = readNoble(node);
          return noble ? {{ noble }} : {{ color: String(node.getAttribute("data-color") || "") }};
        }}),
      }}
    : {{ kind: "none", options: [] }};

  const turnRoot = document.querySelector(cfg.turnIndicatorRoot);
  const turnSeat = String(turnRoot?.getAttribute("data-seat") || "P0");

  return {{
    players: {{
      P0: readPlayer("P0"),
      P1: readPlayer("P1"),
    }},
    bank: readCountMap(document.querySelector(cfg.bankRoot), "token-color"),
    faceup,
    nobles: Array.from(document.querySelectorAll(`${{cfg.nobleRoot}} [data-noble]`)).map(readNoble).filter(Boolean),
    current_turn_seat: turnSeat,
    modal,
    animations_active: document.querySelector(cfg.animationSelector) !== null,
  }};
}})()
"""
