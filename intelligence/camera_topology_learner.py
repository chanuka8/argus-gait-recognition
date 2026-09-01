import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from monitoring.logging_config import get_logger


@dataclass
class LearnedEdgeStats:
    source_camera: str
    destination_camera: str
    transition_count: int = 0
    travel_time_samples: list[float] = field(default_factory=list)
    mean_travel_time: float = 0.0
    median_travel_time: float = 0.0
    robust_lower_bound: float = 0.0
    robust_upper_bound: float = 0.0
    learned_transition_probability: float = 0.0
    last_update_time: float = 0.0


class CameraTopologyLearner:
    def __init__(self, config: dict[str, Any] | None = None, transition_model: Any | None = None) -> None:
        self.logger = get_logger("camera_topology_learner")
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.shadow_mode = bool(cfg.get("shadow_mode", True))
        self.minimum_samples = int(cfg.get("minimum_samples", 20))
        self.maximum_travel_seconds = float(cfg.get("maximum_travel_seconds", 600.0))
        self.sync_interval_seconds = float(cfg.get("sync_interval_seconds", 30.0))
        self.export_path = str(cfg.get("export_path", "outputs/reports/exports/learned_camera_topology.yaml"))
        self.transition_model = transition_model
        self.last_sync_time = -float("inf")

        self.learned_edges: dict[tuple[str, str], LearnedEdgeStats] = {}
        self.exit_events: dict[tuple[str, str], tuple[float, float, float]] = {}

    def set_transition_model(self, transition_model: Any) -> None:
        self.transition_model = transition_model

    def is_enabled(self) -> bool:
        return self.enabled

    def record_camera_exit(
        self,
        camera_id: str,
        identity: str,
        reliability: float,
        occlusion: float,
        timestamp: float | None = None,
    ) -> None:
        if not self.enabled or identity == "UNKNOWN":
            return

        now = timestamp if timestamp is not None else time.monotonic()
        key = (camera_id, identity)
        self.exit_events[key] = (now, float(reliability), float(occlusion))

    def observe_transition(
        self,
        source_camera: str,
        destination_camera: str,
        identity: str,
        reliability: float,
        occlusion: float,
        is_known_identity: bool = True,
        is_temporally_confirmed: bool = True,
        timestamp: float | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        now = timestamp if timestamp is not None else time.monotonic()

        if source_camera == destination_camera:
            return False

        if (
            not is_known_identity
            or not is_temporally_confirmed
            or identity == "UNKNOWN"
            or reliability < 0.70
            or occlusion >= 0.35
        ):
            return False

        exit_key = (source_camera, identity)
        if exit_key not in self.exit_events:
            return False

        exit_time, exit_rel, exit_occ = self.exit_events[exit_key]
        travel_time = now - exit_time

        if travel_time < 0.5 or travel_time > self.maximum_travel_seconds:
            return False

        if exit_rel < 0.70 or exit_occ >= 0.35:
            return False

        edge_key = (source_camera, destination_camera)
        if edge_key not in self.learned_edges:
            self.learned_edges[edge_key] = LearnedEdgeStats(
                source_camera=source_camera,
                destination_camera=destination_camera,
            )

        edge = self.learned_edges[edge_key]
        edge.transition_count += 1
        edge.travel_time_samples.append(float(travel_time))

        if len(edge.travel_time_samples) > 100:
            edge.travel_time_samples.pop(0)

        samples = np.array(edge.travel_time_samples, dtype=np.float32)
        edge.mean_travel_time = float(np.mean(samples))
        edge.median_travel_time = float(np.median(samples))

        if len(samples) >= 3:
            std = float(np.std(samples))
            edge.robust_lower_bound = float(max(0.5, edge.mean_travel_time - 2.0 * std))
            edge.robust_upper_bound = float(min(self.maximum_travel_seconds, edge.mean_travel_time + 2.0 * std))
        else:
            edge.robust_lower_bound = float(max(0.5, edge.mean_travel_time * 0.5))
            edge.robust_upper_bound = float(min(self.maximum_travel_seconds, edge.mean_travel_time * 1.5))

        edge.last_update_time = now

        self._update_transition_probabilities(source_camera)

        return True

    def _update_transition_probabilities(self, source_camera: str) -> None:
        src_edges = [edge for (src, _), edge in self.learned_edges.items() if src == source_camera]
        total_count = sum(edge.transition_count for edge in src_edges)

        if total_count > 0:
            for edge in src_edges:
                edge.learned_transition_probability = float(edge.transition_count / total_count)

    def get_suggested_topology(self) -> dict[str, Any]:
        suggestions = {}
        for (src, dst), edge in self.learned_edges.items():
            if edge.transition_count >= self.minimum_samples:
                key = f"{src}_to_{dst}"
                suggestions[key] = {
                    "source_camera": src,
                    "destination_camera": dst,
                    "transition_count": edge.transition_count,
                    "learned_probability": round(edge.learned_transition_probability, 4),
                    "mean_travel_seconds": round(edge.mean_travel_time, 2),
                    "min_travel_seconds": round(edge.robust_lower_bound, 2),
                    "max_travel_seconds": round(edge.robust_upper_bound, 2),
                }
        return suggestions

    def export_learned_topology(self, output_path: str | None = None) -> str:
        target_path = Path(output_path or self.export_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "version": "1.0",
            "shadow_mode": self.shadow_mode,
            "minimum_samples": self.minimum_samples,
            "maximum_travel_seconds": self.maximum_travel_seconds,
            "suggested_transitions": {},
        }

        for (src, dst), edge in self.learned_edges.items():
            if edge.transition_count >= self.minimum_samples:
                key = f"{src}_to_{dst}"
                export_data["suggested_transitions"][key] = {
                    "source_camera": src,
                    "destination_camera": dst,
                    "transition_count": edge.transition_count,
                    "learned_probability": round(edge.learned_transition_probability, 4),
                    "mean_travel_seconds": round(edge.mean_travel_time, 2),
                    "median_travel_seconds": round(edge.median_travel_time, 2),
                    "min_travel_seconds": round(edge.robust_lower_bound, 2),
                    "max_travel_seconds": round(edge.robust_upper_bound, 2),
                }

        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        return str(target_path)

    def load_learned_topology(self, input_path: str | None = None) -> bool:
        target_path = Path(input_path or self.export_path)
        if not target_path.exists():
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            transitions = data.get("suggested_transitions", {})
            for tdata in transitions.values():
                src = tdata.get("source_camera")
                dst = tdata.get("destination_camera")
                if not src or not dst:
                    continue
                edge_key = (src, dst)
                count = int(tdata.get("transition_count", 0))
                prob = float(tdata.get("learned_probability", 0.0))
                mean_t = float(tdata.get("mean_travel_seconds", 0.0))
                median_t = float(tdata.get("median_travel_seconds", mean_t))
                min_t = float(tdata.get("min_travel_seconds", 0.5))
                max_t = float(tdata.get("max_travel_seconds", self.maximum_travel_seconds))

                self.learned_edges[edge_key] = LearnedEdgeStats(
                    source_camera=src,
                    destination_camera=dst,
                    transition_count=count,
                    travel_time_samples=[mean_t] * min(count, 10),
                    mean_travel_time=mean_t,
                    median_travel_time=median_t,
                    robust_lower_bound=min_t,
                    robust_upper_bound=max_t,
                    learned_transition_probability=prob,
                )
            return True
        except (yaml.YAMLError, OSError, ValueError, KeyError) as e:
            self.logger.warning(f"Failed to load learned topology from {target_path}: {e}")
            return False

    def update_transition_model(self, transition_model: Any | None = None) -> int:
        target = transition_model or self.transition_model
        if target is None or self.shadow_mode or not self.enabled:
            return 0

        updated_count = 0
        for (src, dst), edge in self.learned_edges.items():
            if edge.transition_count >= self.minimum_samples and hasattr(target, "add_or_update_rule"):
                target.add_or_update_rule(
                    source_camera=src,
                    destination_camera=dst,
                    min_travel_seconds=edge.robust_lower_bound,
                    max_travel_seconds=edge.robust_upper_bound,
                    probability=edge.learned_transition_probability,
                )
                updated_count += 1
        return updated_count

    def maybe_sync_transition_model(
        self,
        transition_model: Any | None = None,
        timestamp: float | None = None,
    ) -> int:
        if self.shadow_mode or not self.enabled:
            return 0

        target = transition_model or self.transition_model
        if target is None:
            return 0

        now = timestamp if timestamp is not None else time.monotonic()
        if (now - self.last_sync_time) < self.sync_interval_seconds:
            return 0

        count = self.update_transition_model(target)
        self.last_sync_time = now
        return count

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "CameraTopologyLearner":
        return cls(config=config)

    def reset(self) -> None:
        self.learned_edges.clear()
        self.exit_events.clear()
        self.last_sync_time = -float("inf")

    def cleanup_inactive(self, max_idle_seconds: float = 3600.0, current_time: float | None = None) -> None:
        now = current_time if current_time is not None else time.monotonic()
        for key, (ts, _, _) in list(self.exit_events.items()):
            if (now - ts) > max_idle_seconds:
                del self.exit_events[key]
