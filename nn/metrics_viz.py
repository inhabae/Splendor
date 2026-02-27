from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS_CSV_COLUMNS = (
    "timestamp",
    "mode",
    "run_id",
    "cycle",
    "global_step",
    "axis_cycle",
    "axis_step",
    "metric",
    "value",
)


@dataclass
class MetricRecord:
    timestamp: str
    mode: str
    run_id: str
    cycle: int
    global_step: int
    axis_cycle: float
    axis_step: float
    metric: str
    value: float

    def to_csv_row(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "run_id": self.run_id,
            "cycle": self.cycle,
            "global_step": self.global_step,
            "axis_cycle": self.axis_cycle,
            "axis_step": self.axis_step,
            "metric": self.metric,
            "value": self.value,
        }


class MetricsVizLogger:
    def __init__(
        self,
        *,
        mode: str,
        run_id: str,
        root_dir: str,
        run_name: str,
        save_every_cycle: int = 1,
    ) -> None:
        if save_every_cycle <= 0:
            raise ValueError("save_every_cycle must be positive")
        self.mode = str(mode)
        self.run_id = str(run_id)
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.run_name = str(run_name)
        self.save_every_cycle = int(save_every_cycle)
        self.run_dir = self.root_dir / self.run_name
        self.tb_dir = self.run_dir / "tb"
        self.plots_dir = self.run_dir / "plots"
        self.csv_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "metrics_summary.json"
        self.records: list[MetricRecord] = []
        self.non_finite_count = 0

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Visualization requires matplotlib. Install with: pip install matplotlib"
            ) from exc
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise RuntimeError(
                "Visualization requires tensorboard support via torch.utils.tensorboard. "
                "Install with: pip install tensorboard"
            ) from exc

        self._plt = plt
        self._summary_writer = SummaryWriter
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.writer = self._summary_writer(log_dir=str(self.tb_dir))
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(METRICS_CSV_COLUMNS))
            writer.writeheader()

    def log_scalar(
        self,
        metric: str,
        value: float,
        *,
        cycle: int,
        global_step: int,
        axis_cycle: float,
        axis_step: float,
    ) -> None:
        scalar = float(value)
        if not math.isfinite(scalar):
            self.non_finite_count += 1
        ts = datetime.now(timezone.utc).isoformat()
        record = MetricRecord(
            timestamp=ts,
            mode=self.mode,
            run_id=self.run_id,
            cycle=int(cycle),
            global_step=int(global_step),
            axis_cycle=float(axis_cycle),
            axis_step=float(axis_step),
            metric=str(metric),
            value=scalar,
        )
        self.records.append(record)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(METRICS_CSV_COLUMNS))
            writer.writerow(record.to_csv_row())
        self.writer.add_scalar(str(metric), scalar, int(global_step))

    def maybe_save(self, cycle: int) -> None:
        if cycle <= 0:
            return
        if cycle % self.save_every_cycle != 0:
            return
        self._write_plots()
        self._write_summary()

    def finalize(self) -> None:
        self._write_plots()
        self._write_summary()
        self.writer.flush()
        self.writer.close()

    def _write_plots(self) -> None:
        self._plot_multi(
            title="Loss Curves (by global train step)",
            axis="axis_step",
            metric_names=(
                "train/policy_loss_step",
                "train/value_loss_step",
                "train/total_loss_step",
                "eval/policy_loss_full_replay",
                "eval/value_loss_full_replay",
                "eval/total_loss_full_replay",
            ),
            x_label="Global train step",
            output_path=self.plots_dir / "loss_by_step.png",
        )
        self._plot_multi(
            title="Loss Curves (by cycle)",
            axis="axis_cycle",
            metric_names=(
                "train/policy_loss_cycle_avg",
                "train/value_loss_cycle_avg",
                "train/total_loss_cycle_avg",
                "eval/policy_loss_full_replay",
                "eval/value_loss_full_replay",
                "eval/total_loss_full_replay",
            ),
            x_label="Cycle",
            output_path=self.plots_dir / "loss_by_cycle.png",
        )
        self._plot_multi(
            title="Human Metrics (by global train step)",
            axis="axis_step",
            metric_names=(
                "train/action_top1_acc_step",
                "train/value_sign_acc_step",
                "train/value_mae_step",
                "eval/action_top1_acc",
                "eval/value_sign_acc",
                "eval/value_mae",
            ),
            x_label="Global train step",
            output_path=self.plots_dir / "human_metrics_by_step.png",
        )
        self._plot_multi(
            title="Human Metrics (by cycle)",
            axis="axis_cycle",
            metric_names=(
                "eval/action_top1_acc",
                "eval/value_sign_acc",
                "eval/value_mae",
            ),
            x_label="Cycle",
            output_path=self.plots_dir / "human_metrics_by_cycle.png",
        )

    def _plot_multi(
        self,
        *,
        title: str,
        axis: str,
        metric_names: tuple[str, ...],
        x_label: str,
        output_path: Path,
    ) -> None:
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = 0
        for metric in metric_names:
            points = self._series(metric, axis)
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=metric)
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.3)
        if plotted > 0:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _series(self, metric: str, axis: str) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for record in self.records:
            if record.metric != metric:
                continue
            x = record.axis_step if axis == "axis_step" else record.axis_cycle
            points.append((float(x), float(record.value)))
        points.sort(key=lambda t: t[0])
        return points

    def _write_summary(self) -> None:
        by_metric: dict[str, dict[str, Any]] = {}
        for record in self.records:
            stats = by_metric.get(record.metric)
            if stats is None:
                by_metric[record.metric] = {
                    "count": 1,
                    "last": record.value,
                    "min": record.value,
                    "max": record.value,
                }
                continue
            stats["count"] = int(stats["count"]) + 1
            stats["last"] = float(record.value)
            stats["min"] = min(float(stats["min"]), float(record.value))
            stats["max"] = max(float(stats["max"]), float(record.value))

        payload = {
            "mode": self.mode,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "records": len(self.records),
            "non_finite_values": int(self.non_finite_count),
            "metrics": by_metric,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
