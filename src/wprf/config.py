from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

Offset = Tuple[int, int]  # (dy, dx) in Ω grid (row-major)


def _neighborhood_8() -> Tuple[Offset, ...]:
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


DEFAULT_NEIGHBORHOOD_8: Tuple[Offset, ...] = _neighborhood_8()


def validate_offsets(offsets: Iterable[Offset]) -> Tuple[Offset, ...]:
    offsets_tuple = tuple((int(dy), int(dx)) for dy, dx in offsets)
    if not offsets_tuple:
        raise ValueError("邻域偏移集合 O 不能为空")
    if (0, 0) in offsets_tuple:
        raise ValueError("邻域偏移集合 O 不允许包含 (0,0)")
    if len(set(offsets_tuple)) != len(offsets_tuple):
        raise ValueError("邻域偏移集合 O 不允许包含重复元素")
    missing = sorted({(-dy, -dx) for dy, dx in offsets_tuple} - set(offsets_tuple))
    if missing:
        raise ValueError(f"邻域偏移集合 O 必须关于原点对称，缺少: {missing}")
    return offsets_tuple


@dataclass(frozen=True, slots=True)
class WPRFConstants:
    """
    METHOD.md 中的固定常数集中定义（避免散落魔法常数）。

    注意：
    - `neighborhood_offsets` 在 METHOD.md 中要求“固定且对称”，但具体取值依实验设定；
      这里提供最小默认 `DEFAULT_NEIGHBORHOOD_8` 以便 smoke 运行。
    """

    phi_binarize_threshold: float = 0.5
    phi_l_prune: int = 0
    neighborhood_offsets: Tuple[Offset, ...] = DEFAULT_NEIGHBORHOOD_8
    grid_stride: int = 1

    # METHOD.md 3：自环常数（保证马尔可夫链度数处处 > 0）
    self_loop_lambda: float = 1.0
    self_loop_epsilon0: float = 1.0e-6

    # METHOD.md 4.3 / 5：log 数值稳定项 ε
    log_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if not (0.0 < float(self.phi_binarize_threshold) < 1.0):
            raise ValueError(
                f"phi_binarize_threshold 必须在 (0,1) 内，当前={self.phi_binarize_threshold}"
            )
        if int(self.phi_l_prune) < 0:
            raise ValueError(f"phi_l_prune 必须是非负整数，当前={self.phi_l_prune}")
        validate_offsets(self.neighborhood_offsets)
        if int(self.grid_stride) < 1:
            raise ValueError(f"grid_stride 必须是正整数，当前={self.grid_stride}")

        if float(self.self_loop_lambda) <= 0.0:
            raise ValueError(f"self_loop_lambda 必须满足 >0，当前={self.self_loop_lambda}")
        if float(self.self_loop_epsilon0) <= 0.0:
            raise ValueError(f"self_loop_epsilon0 必须满足 >0，当前={self.self_loop_epsilon0}")
        if float(self.log_epsilon) <= 0.0:
            raise ValueError(f"log_epsilon 必须满足 >0，当前={self.log_epsilon}")
