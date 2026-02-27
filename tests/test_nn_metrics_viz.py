#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import matplotlib.pyplot as _plt  # noqa: F401
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter  # noqa: F401
    _HAS_TB = True
except Exception:  # pragma: no cover
    _HAS_TB = False

if np is not None and torch is not None:
    from nn.metrics_viz import MetricsVizLogger
    from nn.replay import ReplayBuffer, ReplaySample
    from nn.state_schema import ACTION_DIM, STATE_DIM
    from nn.train import _evaluate_on_replay_full
else:
    MetricsVizLogger = None
    ReplayBuffer = None
    ReplaySample = None
    ACTION_DIM = 69
    STATE_DIM = 246
    _evaluate_on_replay_full = None


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestEvalMetrics(unittest.TestCase):
    def test_evaluate_on_replay_full_adds_human_metrics(self):
        class _FixedEvalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                logits = torch.full((3, ACTION_DIM), -5.0, dtype=torch.float32)
                logits[0, 1] = 5.0
                logits[1, 3] = 5.0
                logits[2, 4] = 5.0
                values = torch.tensor([0.8, 0.4, 0.2], dtype=torch.float32)
                self.register_buffer("logits_table", logits)
                self.register_buffer("value_table", values)

            def forward(self, x):
                ids = x[:, 0].long()
                return self.logits_table[ids], self.value_table[ids]

        replay = ReplayBuffer()
        for idx, action, legal, value_target in (
            (0, 1, [0, 1], 1.0),
            (1, 2, [2, 3], -1.0),
            (2, 4, [4], 0.0),
        ):
            state = np.zeros((STATE_DIM,), dtype=np.float32)
            state[0] = float(idx)
            mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
            mask[legal] = True
            policy = np.zeros((ACTION_DIM,), dtype=np.float32)
            policy[action] = 1.0
            replay.add(
                ReplaySample(
                    state=state,
                    mask=mask,
                    action_target=action,
                    value_target=value_target,
                    policy_target=policy,
                )
            )

        metrics = _evaluate_on_replay_full(_FixedEvalModel(), replay, device="cpu")
        self.assertAlmostEqual(metrics["eval_action_top1_acc"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["eval_value_sign_acc"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["eval_value_mae"], 0.6, places=6)


@unittest.skipIf(MetricsVizLogger is None, "visualization module not importable")
@unittest.skipIf(not (_HAS_MPL and _HAS_TB), "visualization runtime dependencies not installed")
class TestMetricsVizArtifacts(unittest.TestCase):
    def test_logger_emits_csv_summary_plots_and_tb_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsVizLogger(
                mode="cycles",
                run_id="run_1",
                root_dir=tmpdir,
                run_name="test_run",
                save_every_cycle=1,
            )
            logger.log_scalar("train/policy_loss_step", 1.2, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/value_loss_step", 0.5, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/total_loss_step", 1.7, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/grad_norm_step", 0.2, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/action_top1_acc_step", 0.3, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/value_sign_acc_step", 0.4, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/value_mae_step", 0.7, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/policy_loss_cycle_avg", 1.0, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/value_loss_cycle_avg", 0.4, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("train/total_loss_cycle_avg", 1.4, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("eval/policy_loss_full_replay", 1.1, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("eval/value_loss_full_replay", 0.6, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("eval/total_loss_full_replay", 1.7, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("eval/samples_full_replay", 42.0, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("eval/action_top1_acc", 0.55, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("eval/value_sign_acc", 0.60, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.log_scalar("eval/value_mae", 0.44, cycle=1, global_step=1, axis_cycle=1.0, axis_step=1.0)
            logger.finalize()

            run_dir = Path(tmpdir) / "test_run"
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "metrics_summary.json").exists())
            self.assertTrue((run_dir / "plots" / "loss_by_step.png").exists())
            self.assertTrue((run_dir / "plots" / "loss_by_cycle.png").exists())
            self.assertTrue((run_dir / "plots" / "human_metrics_by_step.png").exists())
            self.assertTrue((run_dir / "plots" / "human_metrics_by_cycle.png").exists())
            tb_files = list((run_dir / "tb").glob("events.out.tfevents.*"))
            self.assertGreaterEqual(len(tb_files), 1)


if __name__ == "__main__":
    unittest.main()
