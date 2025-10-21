# Subdivision Surface Primitive Overview

> 本文档详细解释了最近一次提交中用细分曲面替换 Bézier 三角面的改动，并记录关键实现细节。

## 改动背景

- **动机**：在训练过程中我们发现高阶 Bézier 三角形的数值稳定性会变差，尤其是在需要求导和做多次重参数化时。细分曲面可以把复杂的曲面评估拆解成局部的线性插值，同时仍然保持 C¹ 连续性，因此更加稳定。
- **接口兼容性**：所有脚本依旧通过 `model.__init__` 导入 `BPrimitiveBezier`。为了避免用户手动修改代码，我们让该名称继续指向新的 `BPrimitiveSubdivision` 类。

## 关键实现概览

1. **控制网格**：`control_point_uvw` 保存了参考三角形上规则采样的重心坐标；学习到的控制点 `self.control_point` 与之对齐。
2. **细分拓扑**：`generate_regular_face` 会在给定阶数下生成固定的子三角拓扑，供渲染、导出时复用。
3. **快速定位**：`_prepare_face_lookup` 预计算每个子三角的逆边矩阵，用于把任意 `(u, v)` 坐标快速映射到局部重心坐标，并顺带得到导数。
4. **缓存**：针对不同分辨率的 mesh 导出需求，`face_cache` 和 `uvw_cache` 预先缓存所有规则网格，避免重复生成。

## 训练/渲染流程影响

- 训练脚本 `demo.py`、`test.py`、`viewer.py` 无需改动；它们依旧调用 `BPrimitiveBezier`，最终会实例化新的细分曲面类。
- `generate_regular_mesh` 现在会直接调用细分评估器，导出的顶点和三角形与旧实现保持顺序一致，保证与已有的 PLY/OBJ 导出脚本兼容。
- 所有一阶导数接口（`evaluate_u_derivative`, `evaluate_v_derivative`, `evaluate_normal`）仍然可用，并且因为有显式缓存，数值更稳定。

## 与旧实现的差异总结

| 维度 | 旧版 Bézier 实现 | 新版细分实现 |
| --- | --- | --- |
| 几何评估 | 直接评估高阶多项式 | 定位子三角后做线性插值 |
| 导数 | 通过 Bézier 基函数求导 | 通过局部重心坐标导数 | 
| 缓存 | 仅缓存随机初始化结构 | 额外缓存子三角拓扑与重心坐标 |
| 接口 | `BPrimitiveBezier` | `BPrimitiveSubdivision`（并保留别名） |

## 未来工作

- 如果需要支持 Catmull–Clark 等非均匀细分规则，可以在 `_prepare_face_lookup` 里拓展不同的拓扑模板。
- 目前缓存的数量是 100，可根据显存情况调整。

