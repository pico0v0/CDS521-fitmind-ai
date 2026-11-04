"""FitMind AI 对比实验批量运行脚本
评测两种方法在 30 组模拟用户与 15 条动态修改指令下的表现：
- 方法 A (Baseline): Direct LLM（无外部候选池过滤，无规则修复）
- 方法 B (Proposed): FitMind AI（候选过滤 + 结构化生成 + 确定性校验与自动修复）
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.schemas import UserProfile, EditIntent
from src.planner_llm import generate_weekly_plan_llm
from src.planner_baseline import generate_weekly_plan_baseline
from src.replanner import apply_plan_edit, parse_intent_hybrid
from src.validator import validate_weekly_plan
from src.retrieval import load_catalog

RESULTS_DIR = ROOT_DIR / "results" / "predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    exp_dir = ROOT_DIR / "experiments"
    profiles = [json.loads(line) for line in (exp_dir / "profiles.jsonl").read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    edits = [json.loads(line) for line in (exp_dir / "edits.jsonl").read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    return profiles, edits


def run_single_initial(profile: dict, method: str, model: str | None = None) -> dict:
    t0 = time.time()
    try:
        if method == "fitmind_ai":
            plan, meta = generate_weekly_plan_llm(profile, model=model)
        else:
            plan, meta = generate_weekly_plan_baseline(profile, model=model)
            
        plan_dict = plan.model_dump()
        val = validate_weekly_plan(plan_dict, profile)
        return {
            "profile_id": profile.get("id"),
            "method": method,
            "status": "success",
            "plan": plan_dict,
            "is_valid": val.is_valid,
            "issues": val.issues,
            "latency_ms": meta.get("latency_ms", int((time.time() - t0) * 1000)),
            "tokens_in": meta.get("tokens_in", 0),
            "tokens_out": meta.get("tokens_out", 0),
            "repaired": meta.get("repaired", False)
        }
    except Exception as e:
        return {
            "profile_id": profile.get("id"),
            "method": method,
            "status": "error",
            "error_msg": str(e),
            "is_valid": False,
            "issues": [f"RuntimeError: {e}"],
            "latency_ms": int((time.time() - t0) * 1000),
            "tokens_in": 0,
            "tokens_out": 0,
            "repaired": False
        }


def run_single_edit(initial_record: dict, edit: dict, method: str, model: str | None = None) -> dict:
    if initial_record.get("status") != "success":
        return {
            "edit_id": edit["id"],
            "profile_id": initial_record.get("profile_id"),
            "method": method,
            "status": "skipped_due_to_initial_failure",
            "edit_success": False,
            "retention_rate": 0.0,
            "issues": ["初始计划生成失败，无法评估编辑"]
        }
    
    plan = initial_record["plan"]
    instruction = edit["instruction"]
    _, foods, _ = load_catalog()
    allowed = [{"id": f["id"], "name": f["name"]} for f in foods]

    t0 = time.time()
    try:
        if method == "fitmind_ai":
            # FitMind AI 使用精准局部重排器
            intent = parse_intent_hybrid(instruction, allowed, model=model)
            updated_plan, summary = apply_plan_edit(plan, intent)
            val = validate_weekly_plan(updated_plan)
            return {
                "edit_id": edit["id"],
                "profile_id": initial_record.get("profile_id"),
                "method": method,
                "status": "success",
                "edit_success": val.is_valid,
                "retention_rate": summary.get("retention_rate", 1.0),
                "issues": val.issues,
                "latency_ms": int((time.time() - t0) * 1000)
            }
        else:
            # Baseline 直接将全部文本交给大模型重新生成一份完整计划
            p_modified = copy_and_modify_profile_text(initial_record["plan"]["profile"], instruction)
            updated_plan, meta = generate_weekly_plan_baseline(p_modified, model=model)
            val = validate_weekly_plan(updated_plan)
            # 计算无差别重写导致的变化元素数
            retention = compute_naive_retention(plan, updated_plan.model_dump())
            return {
                "edit_id": edit["id"],
                "profile_id": initial_record.get("profile_id"),
                "method": method,
                "status": "success",
                "edit_success": val.is_valid,
                "retention_rate": retention,
                "issues": val.issues,
                "latency_ms": meta.get("latency_ms", int((time.time() - t0) * 1000))
            }
    except Exception as e:
        return {
            "edit_id": edit["id"],
            "profile_id": initial_record.get("profile_id"),
            "method": method,
            "status": "error",
            "edit_success": False,
            "retention_rate": 0.0,
            "issues": [f"EditError: {e}"],
            "latency_ms": int((time.time() - t0) * 1000)
        }


def copy_and_modify_profile_text(profile: dict, instruction: str) -> dict:
    p = dict(profile)
    p["notes"] = f"用户最新指令：{instruction}"
    return p


def compute_naive_retention(old_plan: dict, new_plan: dict) -> float:
    """计算两次计划之间未变动的元素比例"""
    total = 7 * 4
    same = 0
    old_week = {d["day"]: d for d in old_plan.get("week", [])}
    new_week = {d["day"]: d for d in new_plan.get("week", [])}
    for day in ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]:
        o_d = old_week.get(day, {})
        n_d = new_week.get(day, {})
        if o_d.get("workout") == n_d.get("workout"):
            same += 1
        o_m = o_d.get("meals", {})
        n_m = n_d.get("meals", {})
        for s in ("早餐", "午餐", "晚餐"):
            if o_m.get(s) == n_m.get(s):
                same += 1
    return round(same / total, 4)


def main():
    parser = argparse.ArgumentParser(description="FitMind AI 对比实验评测")
    parser.add_argument("--limit", type=int, default=0, help="限制评测样本数 (0 表示全部)")
    parser.add_argument("--edits_limit", type=int, default=0, help="限制评测编辑指令数 (0 表示全部)")
    parser.add_argument("--model", type=str, default=None, help="覆盖模型名称")
    args = parser.parse_args()

    profiles, edits = load_dataset()
    if args.limit > 0:
        profiles = profiles[:args.limit]
    if args.edits_limit > 0:
        edits = edits[:args.edits_limit]

    print(f"=== 开始运行对比实验 ===")
    print(f"样本画像数: {len(profiles)} | 编辑指令数: {len(edits)}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for method in ("fitmind_ai", "baseline_direct_llm"):
        print(f"\n>>> 正在运行方法: {method} ...")
        out_file = RESULTS_DIR / f"{method}_{timestamp}.jsonl"
        
        initial_records = []
        for i, p in enumerate(profiles, 1):
            print(f"  [{i}/{len(profiles)}] 初始计划生成 Profile: {p.get('id')} ...", end="", flush=True)
            res = run_single_initial(p, method, model=args.model)
            print(f" {'通过' if res['is_valid'] else '违规 (' + str(len(res['issues'])) + ')'} (耗时 {res['latency_ms']}ms)")
            initial_records.append(res)
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": "initial", **res}, ensure_ascii=False) + "\n")

        print(f"\n>>> 正在运行方法 {method} 的局部编辑测试 ...")
        for j, edit in enumerate(edits, 1):
            # 将编辑测试应用到对应的测试画像（轮转对应）
            base_rec = initial_records[(j - 1) % len(initial_records)]
            print(f"  [{j}/{len(edits)}] 编辑指令 {edit['id']}: '{edit['instruction']}' on {base_rec['profile_id']} ...", end="", flush=True)
            edit_res = run_single_edit(base_rec, edit, method, model=args.model)
            print(f" {'成功' if edit_res['edit_success'] else '失败'} (保留率: {edit_res.get('retention_rate', 0)*100:.1f}%)")
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": "edit", **edit_res}, ensure_ascii=False) + "\n")

    print(f"\n=== 实验全部完成！原始数据已保存至 {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
