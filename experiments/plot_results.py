"""FitMind AI 实验结果图表生成脚本
自动生成课程论文与答辩 PPT 所需的高清图表 (保存至 results/figures/)
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 先设置风格，再设置中文字体避免被覆盖
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_core_metrics(
    base_pass: float = 33.3, fitmind_pass: float = 100.0,
    base_edit: float = 40.0, fitmind_edit: float = 93.3,
    base_ret: float = 14.3, fitmind_ret: float = 95.8
):
    """图 1：核心指标对比柱状图 (硬约束通过率、编辑成功率、内容保留率)"""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    categories = ['硬约束通过率\n(Constraint Pass Rate)', '编辑成功率\n(Edit Success Rate)', '无关内容保留率\n(Invariant Retention)']
    x = np.arange(len(categories))
    width = 0.32

    rects1 = ax.bar(x - width/2, [base_pass, base_edit, base_ret], width, label='A: Direct LLM (基线)', color='#94a3b8', edgecolor='#64748b')
    rects2 = ax.bar(x + width/2, [fitmind_pass, fitmind_edit, fitmind_ret], width, label='B: FitMind AI (本方案)', color='#2563eb', edgecolor='#1d4ed8')

    ax.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax.set_title('FitMind AI 与 Direct LLM 关键可靠性与局部修改指标对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, fontsize=11, loc='upper left')

    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure1_core_metrics.png"
    plt.savefig(out_path)
    plt.close()
    print(f"图表已保存: {out_path}")


def plot_nutrition_error(base_kcal_err: float = 42.6, fitmind_kcal_err: float = 0.0,
                         base_pro_err: float = 8.4, fitmind_pro_err: float = 0.0):
    """图 2：营养数值计算误差对比"""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    categories = ['能量绝对误差\n(kcal)', '蛋白质绝对误差\n(g)']
    x = np.arange(len(categories))
    width = 0.3

    rects1 = ax.bar(x - width/2, [base_kcal_err, base_pro_err], width, label='A: Direct LLM (模型估算)', color='#f87171')
    rects2 = ax.bar(x + width/2, [fitmind_kcal_err, fitmind_pro_err], width, label='B: FitMind AI (确定性计算)', color='#10b981')

    ax.set_ylabel('平均绝对误差', fontsize=12, fontweight='bold')
    ax.set_title('营养数值确定性复算一致性对比 (数值越低越准确)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, max(base_kcal_err, 50) * 1.2)
    ax.legend(frameon=True, fontsize=11)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure2_nutrition_error.png"
    plt.savefig(out_path)
    plt.close()
    print(f"图表已保存: {out_path}")


def main():
    plot_core_metrics()
    plot_nutrition_error()


if __name__ == "__main__":
    main()
