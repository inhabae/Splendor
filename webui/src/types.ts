export type Seat = 'P0' | 'P1';
export type JobStatus = 'QUEUED' | 'RUNNING' | 'DONE' | 'FAILED' | 'CANCELLED';

export interface CheckpointDTO {
  id: string;
  name: string;
  path: string;
  created_at: string;
  size_bytes: number;
}

export interface ActionInfoDTO {
  action_idx: number;
  label: string;
}

export interface MoveLogEntryDTO {
  turn_index: number;
  actor: Seat;
  action_idx: number;
  label: string;
}

export interface GameConfigDTO {
  checkpoint_id: string;
  checkpoint_path: string;
  num_simulations: number;
  player_seat: Seat;
  seed: number;
}

export interface ColorCountsDTO {
  white: number;
  blue: number;
  green: number;
  red: number;
  black: number;
}

export interface TokenCountsDTO extends ColorCountsDTO {
  gold: number;
}

export interface CardDTO {
  points: number;
  bonus_color: 'white' | 'blue' | 'green' | 'red' | 'black';
  cost: ColorCountsDTO;
  source: 'faceup' | 'reserved_public';
  tier?: number;
  slot?: number;
}

export interface NobleDTO {
  points: number;
  requirements: ColorCountsDTO;
}

export interface TierRowDTO {
  tier: number;
  deck_count: number;
  cards: CardDTO[];
}

export interface PlayerBoardDTO {
  seat: Seat;
  display_name: string;
  points: number;
  tokens: TokenCountsDTO;
  bonuses: ColorCountsDTO;
  reserved_public: CardDTO[];
  reserved_total: number;
  is_to_move: boolean;
}

export interface BoardStateDTO {
  meta: {
    target_points: number;
    turn_index: number;
    player_to_move: Seat;
  };
  players: [PlayerBoardDTO, PlayerBoardDTO];
  bank: TokenCountsDTO;
  nobles: NobleDTO[];
  tiers: [TierRowDTO, TierRowDTO, TierRowDTO];
}

export interface GameSnapshotDTO {
  game_id: string;
  status: string;
  player_to_move: Seat;
  legal_actions: number[];
  legal_action_details: ActionInfoDTO[];
  phase_flags: {
    is_return_phase: boolean;
    is_noble_choice_phase: boolean;
  };
  winner: number;
  turn_index: number;
  move_log: MoveLogEntryDTO[];
  config?: GameConfigDTO;
  board_state?: BoardStateDTO | null;
}

export interface EngineThinkResponse {
  job_id: string;
  status: 'QUEUED' | 'RUNNING';
}

export interface EngineJobStatusDTO {
  job_id: string;
  status: JobStatus;
  error?: string | null;
  result?: {
    action_idx: number;
  } | null;
}

export interface PlayerMoveResponse {
  snapshot: GameSnapshotDTO;
  engine_should_move: boolean;
}
