"""Core search logic for memory-aware veRL/vLLM configuration tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True, order=True)
class Candidate:
    """Memory-sensitive knobs for a colocated actor/rollout/ref workload."""

    actor_micro_batch: int
    rollout_logprob_micro_batch: int
    ref_logprob_micro_batch: int
    vllm_gpu_memory_utilization: float
    use_dynamic_batching: bool = False
    parameter_offload: bool = False
    optimizer_offload: bool = False

    @property
    def key(self) -> str:
        return (
            f"a{self.actor_micro_batch}-r{self.rollout_logprob_micro_batch}"
            f"-ref{self.ref_logprob_micro_batch}"
            f"-u{self.vllm_gpu_memory_utilization:.2f}"
            f"-dyn{int(self.use_dynamic_batching)}"
            f"-po{int(self.parameter_offload)}-oo{int(self.optimizer_offload)}"
        )

    @property
    def pressure(self) -> float:
        """Monotonic proxy used only to order unmeasured candidates."""

        offload_factor = 0.72 if self.parameter_offload else 1.0
        optimizer_factor = 0.82 if self.optimizer_offload else 1.0
        dynamic_factor = 0.88 if self.use_dynamic_batching else 1.0
        batch_pressure = (
            0.50 * self.actor_micro_batch
            + 0.30 * self.rollout_logprob_micro_batch
            + 0.20 * self.ref_logprob_micro_batch
        )
        return (
            batch_pressure
            * self.vllm_gpu_memory_utilization
            * offload_factor
            * optimizer_factor
            * dynamic_factor
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    """Result of one bounded profiling trial."""

    candidate: Candidate
    status: str
    elapsed_seconds: float
    peak_memory_mib: float | None = None
    throughput_tokens_per_second: float | None = None
    completed_steps: int = 0
    exit_code: int = 0
    failure_kind: str | None = None
    log_path: str | None = None

    @property
    def successful(self) -> bool:
        return self.status == "success" and self.completed_steps > 0

    def to_dict(self) -> dict:
        output = asdict(self)
        output["candidate"] = self.candidate.to_dict()
        return output


@dataclass(frozen=True)
class SearchSpace:
    actor_micro_batches: tuple[int, ...] = (1, 2, 4, 8, 16)
    rollout_logprob_micro_batches: tuple[int, ...] = (2, 4, 8, 16)
    ref_logprob_micro_batches: tuple[int, ...] = (2, 4, 8, 16)
    vllm_gpu_memory_utilizations: tuple[float, ...] = (0.50, 0.60, 0.65, 0.70, 0.75)
    dynamic_batching: tuple[bool, ...] = (False, True)
    parameter_offload: tuple[bool, ...] = (False,)
    optimizer_offload: tuple[bool, ...] = (False,)

    def candidates(
        self,
        *,
        global_batch_size: int,
        world_size: int,
    ) -> list[Candidate]:
        if global_batch_size <= 0 or world_size <= 0:
            raise ValueError("global_batch_size and world_size must be positive")

        per_rank_batch = global_batch_size // world_size
        if per_rank_batch == 0:
            raise ValueError("global batch must be at least world size")

        candidates = []
        for values in product(
            self.actor_micro_batches,
            self.rollout_logprob_micro_batches,
            self.ref_logprob_micro_batches,
            self.vllm_gpu_memory_utilizations,
            self.dynamic_batching,
            self.parameter_offload,
            self.optimizer_offload,
        ):
            item = Candidate(*values)
            micro_batches = (
                item.actor_micro_batch,
                item.rollout_logprob_micro_batch,
                item.ref_logprob_micro_batch,
            )
            if any(value > per_rank_batch for value in micro_batches):
                continue
            if not item.use_dynamic_batching and any(
                per_rank_batch % value != 0 for value in micro_batches
            ):
                continue
            candidates.append(item)
        return sorted(candidates, key=lambda item: (item.pressure, item.key))


class Tuner:
    """Select profiling trials and the best measured feasible configuration."""

    def __init__(
        self,
        candidates: Iterable[Candidate],
        *,
        gpu_memory_mib: float,
        safety_margin: float = 0.08,
    ):
        self.candidates = tuple(sorted(set(candidates), key=lambda item: (item.pressure, item.key)))
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        if gpu_memory_mib <= 0:
            raise ValueError("gpu_memory_mib must be positive")
        if not 0 <= safety_margin < 1:
            raise ValueError("safety_margin must be in [0, 1)")
        self.gpu_memory_mib = gpu_memory_mib
        self.safety_margin = safety_margin

    @property
    def memory_limit_mib(self) -> float:
        return self.gpu_memory_mib * (1.0 - self.safety_margin)

    def choose_next(self, observations: Iterable[Observation]) -> Candidate | None:
        """Probe the unknown memory frontier using pressure-ordered bisection."""

        observations = tuple(observations)
        measured = {item.candidate for item in observations}
        unknown = [candidate for candidate in self.candidates if candidate not in measured]
        if not unknown:
            return None

        successes = [item.candidate.pressure for item in observations if item.successful]
        memory_failures = [
            item.candidate.pressure
            for item in observations
            if item.failure_kind in {"oom", "memory_limit"}
        ]

        lower = max(successes, default=float("-inf"))
        upper = min(memory_failures, default=float("inf"))
        frontier = [candidate for candidate in unknown if lower < candidate.pressure < upper]
        if frontier:
            return frontier[len(frontier) // 2]
        if not successes:
            return unknown[len(unknown) // 2]
        if not memory_failures:
            return unknown[-1]

        boundary = (lower + upper) / 2
        return min(unknown, key=lambda candidate: abs(candidate.pressure - boundary))

    def select_best(self, observations: Iterable[Observation]) -> Observation | None:
        """Return the fastest successful trial respecting the memory margin."""

        feasible = []
        for observation in observations:
            if not observation.successful:
                continue
            if (
                observation.peak_memory_mib is not None
                and observation.peak_memory_mib > self.memory_limit_mib
            ):
                continue
            feasible.append(observation)
        if not feasible:
            return None

        def score(item: Observation) -> tuple[float, float, float]:
            throughput = item.throughput_tokens_per_second or 0.0
            memory = item.peak_memory_mib or self.memory_limit_mib
            return (throughput, -memory, -item.elapsed_seconds)

        return max(feasible, key=score)
