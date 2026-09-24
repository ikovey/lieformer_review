"""Self-contained Sparse Harmonic Recoupling (SHR) implementation."""
from .paths import OuterPath, SHRBranch, RecoupledPath, all_admissible_paths, build_outer_paths, recoupling_coefficient, recouple_path, recouple_paths
from .primitives import center_positions, component_tensor_product, regular_solid_harmonics, scatter_sum
from .shr import SHR, SHRLayer, SHRPlan, SparseHarmonicRecoupling

__all__ = [
    "SHR", "SHRLayer", "SparseHarmonicRecoupling", "SHRPlan",
    "OuterPath", "SHRBranch", "RecoupledPath", "all_admissible_paths", "build_outer_paths",
    "recoupling_coefficient", "recouple_path", "recouple_paths",
    "center_positions", "component_tensor_product", "regular_solid_harmonics", "scatter_sum",
]
