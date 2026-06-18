import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, text, width=2.3, height=0.75, color="#E8F1FA"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.2,
        edgecolor="#2F4B7C",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)
    return box


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2,
        color="#333333",
    )
    ax.add_patch(arrow)


def main():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(workspace_dir, "src", "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    add_box(ax, (0.4, 5.5), "场站特性\n地区/容量/组件效率/积灰", color="#FDEBD0")
    add_box(ax, (0.4, 3.8), "历史时序\n辐照度/温度/功率", color="#E8F8F5")
    add_box(ax, (0.4, 2.1), "未来气象\nNWP 辐照度/温度", color="#E8F8F5")
    add_box(ax, (0.4, 0.4), "运维事件\n检修/故障/限电", color="#FADBD8")

    add_box(ax, (3.4, 5.5), "场站专属建模\nStation-specific model", color="#FDEBD0")
    add_box(ax, (3.4, 3.8), "连续特征映射\nLinear Projection", color="#D6EAF8")
    add_box(ax, (3.4, 0.4), "事件嵌入\nO&M Event Embedding", color="#F5B7B1")

    add_box(ax, (6.2, 4.0), "特征级融合\nConcat + MLP", color="#D7BDE2")
    add_box(ax, (6.2, 2.5), "Transformer 编码器/解码器\nTemporal Attention", width=2.7, color="#D6EAF8")
    add_box(ax, (9.2, 3.6), "原始功率预测\nRaw Prediction", color="#E8F8F5")
    add_box(ax, (9.2, 2.1), "O&M Gate\n可微输出抑制", color="#F5B7B1")
    add_box(ax, (9.2, 0.7), "短期功率预测\nFinal Forecast", color="#D5F5E3")

    add_arrow(ax, (2.7, 5.88), (3.4, 5.88))
    add_arrow(ax, (2.7, 4.18), (3.4, 4.18))
    add_arrow(ax, (2.7, 2.48), (6.2, 4.2))
    add_arrow(ax, (2.7, 0.78), (3.4, 0.78))
    add_arrow(ax, (5.7, 5.88), (6.2, 4.72))
    add_arrow(ax, (5.7, 4.18), (6.2, 4.38))
    add_arrow(ax, (5.7, 0.78), (6.2, 4.02))
    add_arrow(ax, (7.55, 4.0), (7.55, 3.25))
    add_arrow(ax, (8.9, 2.88), (9.2, 3.92))
    add_arrow(ax, (8.9, 2.72), (9.2, 2.42))
    add_arrow(ax, (10.35, 3.6), (10.35, 2.85))
    add_arrow(ax, (10.35, 2.1), (10.35, 1.45))

    ax.text(6.0, 6.55, "融合场站异构性与运维事件感知的 Transformer 光伏电站短期功率预测框架", ha="center", fontsize=14, weight="bold")
    ax.text(5.0, 0.25, "未来故障/临时限电在预测时屏蔽，仅保留可提前获得的计划检修信息，避免信息泄露。", ha="center", fontsize=9, color="#555555")

    out_path = os.path.join(results_dir, "model_architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
