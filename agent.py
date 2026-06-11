
"""
labagent_pro.py — Agent 10/10 : déterministe, robuste, architecturé.
- Façade propre
- Runtime séparé
- Registry typé
- Contexte de pipeline
- Gestion d’erreurs déterministe
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import labutils_pro as lu

T = TypeVar("T")
U = TypeVar("U")


# ============================================================================
# 0. Erreurs agentiques
# ============================================================================

class AgentError(Exception):
    """Erreur générique de l’agent."""


class CapabilityError(AgentError):
    """Erreur lors de l’exécution d’une capacité."""


class PipelineError(AgentError):
    """Erreur lors de l’exécution d’un pipeline."""


class RegistryError(AgentError):
    """Erreur liée au registre de capacités."""


# ============================================================================
# 1. Protocoles & contrats
# ============================================================================

class Capability(Protocol):
    """Contrat minimal pour une capacité agentique."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class PipelineStep(Protocol):
    """Contrat pour une étape de pipeline : f(x, ctx) -> y."""

    def __call__(self, x: Any, ctx: "PipelineContext") -> Any:
        ...


@dataclass(frozen=True)
class AgentConfig:
    seed: int = 1234
    max_workers: Optional[int] = None
    batch_size: int = 32
    fail_fast: bool = True
    log_level: int = logging.INFO
    name: str = "labagent_pro"

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def apply(self) -> logging.Logger:
        """Applique la configuration globale (seed + logging) et retourne le logger."""
        self.validate()
        lu.set_global_seed(self.seed)
        logger = lu.get_logger(self.name, self.log_level)
        logger.info(
            f"AgentConfig applied "
            f"(seed={self.seed}, batch_size={self.batch_size}, fail_fast={self.fail_fast})"
        )
        return logger


# ============================================================================
# 2. Contexte runtime & pipeline
# ============================================================================

@dataclass
class PipelineContext:
    """Contexte partagé entre les étapes de pipeline."""

    step_index: int
    total_steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_next_step(self) -> "PipelineContext":
        return PipelineContext(
            step_index=self.step_index + 1,
            total_steps=self.total_steps,
            metadata=self.metadata,
        )


@dataclass
class AgentRuntime:
    """Runtime déterministe : encapsule les primitives de labutils_pro."""

    config: AgentConfig
    logger: logging.Logger

    def map(
        self,
        fn: Callable[[T], U],
        iterable: Iterable[T],
    ) -> List[U]:
        self.logger.info(
            f"Running deterministic parallel_map "
            f"(len={len(list(iterable)) if hasattr(iterable, '__len__') else 'unknown'})..."
        )
        try:
            return lu.parallel_map(
                fn,
                iterable,
                max_workers=self.config.max_workers,
                batch_size=self.config.batch_size,
                fail_fast=self.config.fail_fast,
            )
        except Exception as e:
            self.logger.error("parallel_map failed", exc_info=True)
            raise AgentError("parallel_map failed") from e

    def benchmark(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict:
        self.logger.info(f"Benchmarking function {getattr(fn, '__name__', str(fn))}...")
        try:
            stats = lu.benchmark(fn, *args, **kwargs)
            self.logger.info(
                f"Benchmark done: mean={stats['mean_ms']:.3f} ms, "
                f"p90={stats['p90_ms']:.3f} ms"
            )
            return stats
        except Exception as e:
            self.logger.error("benchmark failed", exc_info=True)
            raise AgentError("benchmark failed") from e

    def chunks(self, seq: Sequence[T], size: int) -> List[Sequence[T]]:
        self.logger.debug(f"chunks(seq_len={len(seq)}, size={size})")
        try:
            return list(lu.chunks(seq, size))
        except Exception as e:
            self.logger.error("chunks failed", exc_info=True)
            raise AgentError("chunks failed") from e

    def flatten(self, nested: Iterable[Iterable[T]]) -> List[T]:
        self.logger.debug("flatten(...)")
        try:
            return lu.flatten(nested)
        except Exception as e:
            self.logger.error("flatten failed", exc_info=True)
            raise AgentError("flatten failed") from e

    def sliding(self, seq: Sequence[T], k: int) -> List[Tuple[T, ...]]:
        self.logger.debug(f"sliding(seq_len={len(seq)}, k={k})")
        try:
            return list(lu.sliding_window(seq, k))
        except Exception as e:
            self.logger.error("sliding_window failed", exc_info=True)
            raise AgentError("sliding_window failed") from e


# ============================================================================
# 3. Registry typé
# ============================================================================

class AgentRegistry:
    """
    Registry typé de capacités agentiques.
    - noms uniques
    - capacités conformes au protocole Capability
    """

    def __init__(self, logger: logging.Logger):
        self._registry: Dict[str, Capability] = {}
        self._logger = logger

    def register(self, name: str, fn: Capability) -> None:
        if not isinstance(name, str) or not name:
            raise RegistryError("Capability name must be a non-empty string")
        if name in self._registry:
            raise RegistryError(f"Capability '{name}' already registered")
        self._registry[name] = fn
        self._logger.info(f"Registered capability '{name}'")

    def run(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._registry:
            raise RegistryError(f"Unknown capability '{name}'")
        fn = self._registry[name]
        self._logger.info(f"Running capability '{name}'")
        try:
            return fn(*args, **kwargs)
        except AgentError:
            # déjà enrichi, on propage
            raise
        except Exception as e:
            self._logger.error(
                f"Capability '{name}' failed", exc_info=True
            )
            raise CapabilityError(f"Capability '{name}' failed") from e

    def list(self) -> List[str]:
        return sorted(self._registry.keys())


# ============================================================================
# 4. Agent haut niveau (façade)
# ============================================================================

class LabAgent:
    """
    Agent déterministe 10/10 encapsulant labutils_pro.
    - runtime séparé
    - registry typé
    - pipeline contextuel
    """

    def __init__(self, config: AgentConfig):
        logger = config.apply()
        self.config = config
        self.logger = logger
        self.runtime = AgentRuntime(config=config, logger=logger)
        self.registry = AgentRegistry(logger=logger)

        # capacités par défaut
        self._register_default_capabilities()

    # -------------------------------
    # Registration
    # -------------------------------
    def _register_default_capabilities(self) -> None:
        self.registry.register("map", self.map)
        self.registry.register("benchmark", self.benchmark)
        self.registry.register("pipeline", self.pipeline)

    # -------------------------------
    # Façade map
    # -------------------------------
    def map(
        self,
        fn: Callable[[T], U],
        iterable: Iterable[T],
    ) -> List[U]:
        return self.runtime.map(fn, iterable)

    # -------------------------------
    # Façade benchmark
    # -------------------------------
    def benchmark(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict:
        return self.runtime.benchmark(fn, *args, **kwargs)

    # -------------------------------
    # Pipeline contextuel
    # -------------------------------
    def pipeline(
        self,
        steps: List[Callable[[Any, PipelineContext], Any]],
        x: Any,
        metadata: Optional[Dict[str, Any]] = None,
        fail_fast: bool = True,
    ) -> Any:
        """
        Pipeline déterministe avec contexte :
        - chaque étape reçoit (x, ctx)
        - ctx contient index, total, metadata
        - fail_fast configurable
        """
        if not steps:
            raise PipelineError("Pipeline must have at least one step")

        total = len(steps)
        ctx = PipelineContext(step_index=0, total_steps=total, metadata=metadata or {})

        self.logger.info(f"Running pipeline with {total} steps...")
        current = x

        for i, step in enumerate(steps):
            if not callable(step):
                raise PipelineError(f"Step {i} is not callable")

            self.logger.info(
                f"Step {i+1}/{total}: {getattr(step, '__name__', str(step))}"
            )

            try:
                current = step(current, ctx)
            except AgentError:
                # déjà enrichi, on propage
                raise
            except Exception as e:
                self.logger.error(
                    f"Pipeline failed at step {i+1}/{total}", exc_info=True
                )
                if fail_fast:
                    raise PipelineError(
                        f"Pipeline failed at step {i+1}/{total}"
                    ) from e
                # mode non fail_fast : on log, on continue avec la valeur courante
            ctx = ctx.with_next_step()

        return current

    # -------------------------------
    # Helpers exposés
    # -------------------------------
    def chunks(self, seq: Sequence[T], size: int) -> List[Sequence[T]]:
        return self.runtime.chunks(seq, size)

    def flatten(self, nested: Iterable[Iterable[T]]) -> List[T]:
        return self.runtime.flatten(nested)

    def sliding(self, seq: Sequence[T], k: int) -> List[Tuple[T, ...]]:
        return self.runtime.sliding(seq, k)


# ============================================================================
# 5. Builder
# ============================================================================

def build_agent(seed: int = 1234) -> LabAgent:
    config = AgentConfig(seed=seed)
    return LabAgent(config)


# ============================================================================
# 6. Exemple d’utilisation
# ============================================================================

if __name__ == "__main__":
    agent = build_agent(seed=42)

    # Parallel map
    out = agent.registry.run("map", lambda x: x * x, range(10))
    print("map:", out)

    # Benchmark
    stats = agent.registry.run("benchmark", sum, [1, 2, 3, 4, 5])
    print("benchmark:", stats)

    # Pipeline avec contexte
    def step_add(x: int, ctx: PipelineContext) -> int:
        # exemple : on logge dans metadata
        ctx.metadata.setdefault("trace", []).append(
            f"add at step {ctx.step_index}"
        )
        return x + 1

    def step_mul(x: int, ctx: PipelineContext) -> int:
        ctx.metadata.setdefault("trace", []).append(
            f"mul at step {ctx.step_index}"
        )
        return x * 3

    pipeline_out = agent.registry.run(
        "pipeline",
        [step_add, step_mul],
        10,
        metadata={"run_id": "demo"},
    )
    print("pipeline:", pipeline_out)
