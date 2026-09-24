"""Extensible outer-path selection for strict tensor products.

This package deliberately has no dependency on ``src.model.zitp``.  Policies
operate on plain angular triplets and are applied before Wigner-6j recoupling.
"""

from .core import (
    AngularPath,
    OuterPathPolicy,
    PathPolicyRegistry,
    PathSelectionContext,
    build_admissible_candidates,
    build_outer_path_specs,
    path_policy_registry,
    resolve_path_policy,
)
from .policies import ExplicitPathPolicy, FullAdmissiblePolicy, ZippedPolicy
from .probes import OuterPathProbeGates, path_key, path_triplet
from .learned import LearnedOuterPathGates
from .search import (
    PathCandidate,
    TaylorScoreAccumulator,
    enumerate_budget_candidates,
    internal_path_costs,
    random_budget_matched_candidates,
    parse_triplet_token,
    triplet_token,
    write_search_manifest,
)

__all__ = [
    "AngularPath",
    "ExplicitPathPolicy",
    "FullAdmissiblePolicy",
    "OuterPathPolicy",
    "OuterPathProbeGates",
    "LearnedOuterPathGates",
    "PathPolicyRegistry",
    "PathCandidate",
    "PathSelectionContext",
    "ZippedPolicy",
    "TaylorScoreAccumulator",
    "build_admissible_candidates",
    "build_outer_path_specs",
    "enumerate_budget_candidates",
    "internal_path_costs",
    "path_policy_registry",
    "path_key",
    "path_triplet",
    "parse_triplet_token",
    "triplet_token",
    "random_budget_matched_candidates",
    "resolve_path_policy",
    "write_search_manifest",
]
