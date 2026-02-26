"""
WPRF（Widest-Path Reachability Fields）核心模块。

本包严格以 `METHOD.md` 作为唯一算法规格来源，所有算子均在固定建图网格 Ω 上定义。
当前增量提供：
- 确定性的像素域支撑域算子 Φ_px（Zhang–Suen thinning；默认不做端点剥离）
- 像素网格 Ω0 -> 建图网格 Ω 的确定性对齐 Π_s（cell-occupancy / 块最大池化）
- GT 图构建与连通分量标号（实例身份 = 连通分量）
"""

from .config import DEFAULT_NEIGHBORHOOD_8, Offset, WPRFConstants
from .gt import GTGraph, build_gt_graph
from .infer import PredGraph, infer_graph_cc
from .losses import reachability_loss_k
from .markov import WPRFMarkovChain, WPRFMarkovChainTorch, build_markov_chain, build_markov_chain_torch
from .models import SMPUNetWPRF, WPRFFields
from .pairs import MultiScalePairs, sample_multiscale_pairs
from .reachability import discounted_cumulative_reachability, reachability_for_pairs
from .render import (
    RenderedInstances,
    render_instances_voronoi,
    render_instances_voronoi_by_radius_field,
    render_instances_voronoi_image_level,
)
from .phi import phi_support
from .omega import project_bool_to_omega_occupancy, project_nonneg_to_omega_max

__all__ = [
    "DEFAULT_NEIGHBORHOOD_8",
    "GTGraph",
    "Offset",
    "WPRFConstants",
    "WPRFMarkovChain",
    "WPRFMarkovChainTorch",
    "WPRFFields",
    "SMPUNetWPRF",
    "PredGraph",
    "RenderedInstances",
    "MultiScalePairs",
    "build_gt_graph",
    "build_markov_chain",
    "build_markov_chain_torch",
    "discounted_cumulative_reachability",
    "infer_graph_cc",
    "reachability_loss_k",
    "phi_support",
    "project_bool_to_omega_occupancy",
    "project_nonneg_to_omega_max",
    "render_instances_voronoi",
    "render_instances_voronoi_by_radius_field",
    "render_instances_voronoi_image_level",
    "reachability_for_pairs",
    "sample_multiscale_pairs",
]
