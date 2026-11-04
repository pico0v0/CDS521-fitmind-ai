"""FitMind AI — AI 个性化健身与饮食规划系统 (Web 交互演示端)
集成了【FitMind AI 完整生成】、【Direct LLM 对照基线】与【局部高保真重排】。
"""
from __future__ import annotations
import json
import os
import streamlit as st
from pathlib import Path

from src.schemas import UserProfile, EditIntent, WeeklyPlan
from src.retrieval import load_catalog, get_eligible_exercises, get_safe_foods_and_meals
from src.validator import validate_weekly_plan
from src.planner_llm import generate_weekly_plan_llm
from src.planner_baseline import generate_weekly_plan_baseline
from src.core import build_demo_plan
from src.replanner import apply_plan_edit, parse_intent_hybrid

st.set_page_config(
    page_title="FitMind AI — 个性化健身与饮食规划系统",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

DAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
PRESET_PROFILES = {
    "【预设】P01: 徒手/初学减脂 (无过敏)": {
        "age": 21, "height_cm": 161, "weight_kg": 56.0, "goal": "减脂",
        "experience": "初学者", "equipment": "徒手", "sessions": 2,
        "available_days": ["周一", "周三"], "allergies": [], "exclude_foods": []
    },
    "【预设】P05: 健身房/减脂进阶": {
        "age": 25, "height_cm": 165, "weight_kg": 60.0, "goal": "减脂",
        "experience": "初学者", "equipment": "健身房", "sessions": 3,
        "available_days": ["周一", "周三", "周五"], "allergies": [], "exclude_foods": []
    },
    "【预设】P15: 健身房/增肌进阶 (高频次)": {
        "age": 35, "height_cm": 175, "weight_kg": 70.0, "goal": "增肌",
        "experience": "有经验", "equipment": "健身房", "sessions": 4,
        "available_days": ["周一", "周二", "周四", "周六"], "allergies": [], "exclude_foods": []
    },
    "【预设】E01: 乳制品过敏特殊样本": {
        "age": 21, "height_cm": 161, "weight_kg": 56.0, "goal": "减脂",
        "experience": "初学者", "equipment": "徒手", "sessions": 2,
        "available_days": ["周一", "周三"], "allergies": ["乳制品"], "exclude_foods": []
    },
    "【预设】E06: 鸡蛋过敏 + 排除鸡肉与牛肉": {
        "age": 26, "height_cm": 166, "weight_kg": 61.0, "goal": "减脂",
        "experience": "有经验", "equipment": "健身房", "sessions": 2,
        "available_days": ["周一", "周三"], "allergies": ["鸡蛋"], "exclude_foods": ["chicken", "beef"]
    }
}

# 顶部标题栏
st.title("🏋️ FitMind AI｜个性化健身与饮食规划系统")
st.markdown("基于 **大语言模型 (LLM)** 与 **确定性规则约束校验 / 局部修复** 的智能运动与营养决策支持系统。")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统与用户配置")
    
    preset_choice = st.selectbox(
        "快速加载实验画像",
        ["自定义输入"] + list(PRESET_PROFILES.keys())
    )
    
    init_data = PRESET_PROFILES.get(preset_choice, {})
    
    age = st.number_input("年龄 (岁)", 18, 64, int(init_data.get("age", 24)))
    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("身高 (cm)", 130, 220, int(init_data.get("height_cm", 172)))
    with c2:
        weight = st.number_input("体重 (kg)", 30.0, 200.0, float(init_data.get("weight_kg", 66.0)))
        
    goal = st.selectbox("健康塑形目标", ["减脂", "增肌", "塑形", "保持健康"], 
                        index=["减脂", "增肌", "塑形", "保持健康"].index(init_data.get("goal", "减脂")))
    experience = st.selectbox("运动经验", ["初学者", "有经验"],
                              index=["初学者", "有经验"].index(init_data.get("experience", "初学者")))
    equipment = st.selectbox("可用器械等级", ["徒手", "哑铃", "健身房"],
                             index=["徒手", "哑铃", "健身房"].index(init_data.get("equipment", "哑铃")))
    available_days = st.multiselect("允许训练的日期", DAYS, 
                                    default=init_data.get("available_days", ["周一", "周三", "周五"]))
    sessions = st.number_input("每周计划训练次数", 1, max(1, len(available_days)), 
                               min(int(init_data.get("sessions", 3)), len(available_days)))
    
    allergies = st.multiselect("食物过敏类别", ["乳制品", "鸡蛋", "小麦", "大豆", "鱼类", "坚果"],
                               default=init_data.get("allergies", []))
    
    _, all_foods, _ = load_catalog()
    food_id_map = {f["id"]: f["name"] for f in all_foods}
    food_name_map = {f["name"]: f["id"] for f in all_foods}
    
    init_excludes_names = [food_id_map[fid] for fid in init_data.get("exclude_foods", []) if fid in food_id_map]
    exclude_food_names = st.multiselect("主动排除的食材", list(food_name_map.keys()), default=init_excludes_names)
    exclude_food_ids = [food_name_map[name] for name in exclude_food_names]

    st.divider()
    gen_mode = st.radio(
        "🎯 规划生成引擎",
        ["FitMind AI (LLM + 候选检索 + 约束校验修复)", "Direct LLM (端到端基线对照)", "本地确定性规则 Demo"],
        index=0
    )

    gen_btn = st.button("🚀 生成 7 天个性化方案", type="primary", use_container_width=True)

# 处理生成
if gen_btn:
    profile_dict = {
        "age": int(age), "height_cm": int(height), "weight_kg": float(weight),
        "goal": goal, "experience": experience, "equipment": equipment,
        "available_days": available_days, "sessions": int(sessions),
        "allergies": allergies, "exclude_foods": exclude_food_ids
    }
    
    with st.spinner("FitMind AI 正在综合运动科学与营养约束生成方案..."):
        try:
            if gen_mode == "FitMind AI (LLM + 候选检索 + 约束校验修复)":
                plan_obj, meta = generate_weekly_plan_llm(profile_dict)
                st.session_state.current_plan = plan_obj.model_dump()
                st.session_state.plan_meta = meta
            elif gen_mode == "Direct LLM (端到端基线对照)":
                plan_obj, meta = generate_weekly_plan_baseline(profile_dict)
                st.session_state.current_plan = plan_obj.model_dump()
                st.session_state.plan_meta = meta
            else:
                demo_res = build_demo_plan(profile_dict)
                st.session_state.current_plan = demo_res
                st.session_state.plan_meta = {"mode": "local_demo", "latency_ms": 10}
                
            st.session_state.last_diff = None
        except Exception as e:
            st.error(f"生成计划失败: {e}")

# 主体内容展示
if "current_plan" in st.session_state:
    plan = st.session_state.current_plan
    meta = st.session_state.get("plan_meta", {})
    val_result = validate_weekly_plan(plan)

    # 状态条
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        if val_result.is_valid:
            st.success("✅ 硬约束审查：100% 通过")
        else:
            st.error(f"❌ 发现 {len(val_result.issues)} 项规则违规")
    with col_stat2:
        st.metric("生成机制", plan.get("mode", "未知"))
    with col_stat3:
        st.metric("耗时 (Latency)", f"{meta.get('latency_ms', 0)} ms")
    with col_stat4:
        workout_days_cnt = sum(1 for d in plan.get("week", []) if d.get("workout"))
        st.metric("周训练频次", f"{workout_days_cnt} 次 / 周")

    if not val_result.is_valid:
        with st.expander("⚠️ 点击查看违规详情清单", expanded=True):
            for issue in val_result.issues:
                st.warning(issue)

    # 科学性解释卡片
    if plan.get("explanation"):
        with st.expander("💡 查看 AI 科学规划理念与个性化指导建议", expanded=False):
            st.info(plan["explanation"])

    # 局部变动高亮提示
    if st.session_state.get("last_diff"):
        diff = st.session_state.last_diff
        st.success(f"🎉 局部重排成功！无关内容保留率：**{diff.get('unchanged_ratio')}**。变动明细：{'；'.join(diff.get('changes', []))}")

    # 7 天计划详情
    st.markdown("### 📅 7 天训练与餐饮排期总览")
    tabs = st.tabs([f"📌 {d['day']}" for d in plan["week"]])
    
    ex_list, foods_list, _ = load_catalog()
    ex_name_map = {e["id"]: e for e in ex_list}
    food_meta_map = {f["id"]: f for f in foods_list}

    for tab, day_data in zip(tabs, plan["week"]):
        with tab:
            col_workout, col_diet = st.columns([1, 1])
            
            # 左侧：训练卡片
            with col_workout:
                st.subheader("🏃 训练安排")
                w = day_data.get("workout")
                if w:
                    st.markdown(f"**计划时长**：⏱️ {w.get('minutes', 40)} 分钟 | **组数**：{w.get('sets', 3)} 组/动作")
                    if w.get("notes"):
                        st.caption(f"提示：{w['notes']}")
                    
                    st.markdown("**动作清单：**")
                    for eid in w.get("exercise_ids", []):
                        ex_info = ex_name_map.get(eid, {})
                        ex_name = ex_info.get("name", eid)
                        muscle = ex_info.get("muscle", "通用")
                        eq = ex_info.get("equipment", "徒手")
                        desc = ex_info.get("description", "")
                        st.markdown(f"- **{ex_name}** `[{muscle} · {eq}]`  \n  <small style='color: gray;'>{desc}</small>", unsafe_allow_html=True)
                else:
                    st.info("🛋️ **身体恢复与休息日**  \n充分睡眠与肌肉恢复是适应训练负荷的关键。")

            # 右侧：三餐卡片
            with col_diet:
                st.subheader("🥗 膳食营养安排")
                meals = day_data.get("meals", {})
                
                day_totals = {"kcal": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0}
                
                for zh_slot, en_slot in [("早餐", "breakfast"), ("午餐", "lunch"), ("晚餐", "dinner")]:
                    m = meals.get(zh_slot) or meals.get(en_slot)
                    if not m:
                        continue
                    nut = m.get("nutrition", {})
                    for k in day_totals:
                        day_totals[k] += nut.get(k, 0.0)
                    
                    with st.expander(f"**{zh_slot}**：{m.get('name', '定制营养餐')} · {nut.get('kcal', 0):.0f} kcal", expanded=False):
                        st.caption(f"蛋白质: {nut.get('protein', 0):.1f}g | 碳水: {nut.get('carb', 0):.1f}g | 脂肪: {nut.get('fat', 0):.1f}g")
                        st.markdown("**食材克重构成：**")
                        for ing in m.get("ingredients", []):
                            fid = ing.get("food_id") if isinstance(ing, dict) else getattr(ing, "food_id", "")
                            grams = ing.get("grams") if isinstance(ing, dict) else getattr(ing, "grams", 0)
                            fname = food_meta_map.get(fid, {}).get("name", fid)
                            st.write(f"- {fname}: {grams:.0f} g")

                st.markdown(f"**当日摄入总量**：🔥 **{day_totals['kcal']:.0f}** kcal | 🥩 **{day_totals['protein']:.1f}**g 蛋白 | 🍚 **{day_totals['carb']:.1f}**g 碳水 | 🥑 **{day_totals['fat']:.1f}**g 脂肪")

    st.divider()

    # 动态自然语言局部调整区域
    st.markdown("### 💬 交互式自然语言调整 (Replanner)")
    st.caption("支持日常突发调整，系统将在保护其余日程不变（保留率 ≥ 95%）的前提下，精确执行局部变更并自动复检。")
    
    col_input, col_action = st.columns([4, 1])
    with col_input:
        user_edit_text = st.text_input(
            "输入你的修改想法",
            placeholder="例如：周三有事改到周四训练 / 早餐不想吃鸡蛋 / 周五晚餐想吃清淡素食",
            label_visibility="collapsed"
        )
    with col_action:
        edit_submit = st.button("🚀 应用调整", type="secondary", use_container_width=True)

    if edit_submit and user_edit_text.strip():
        with st.spinner("正在解析调整指令并执行高保真局部重排..."):
            try:
                allowed_catalog = [{"id": f["id"], "name": f["name"]} for f in foods_list]
                intent = parse_intent_hybrid(user_edit_text.strip(), allowed_catalog)
                updated_plan, diff_summary = apply_plan_edit(plan, intent)
                st.session_state.current_plan = updated_plan.model_dump()
                st.session_state.last_diff = diff_summary
                st.rerun()
            except Exception as e:
                st.error(f"调整未成功执行: {e}")

    # 下载计划
    st.download_button(
        "📥 下载完整周计划 (JSON)",
        json.dumps(plan, ensure_ascii=False, indent=2),
        file_name=f"fitmind_plan_{plan.get('mode', 'plan')}.json",
        mime="application/json"
    )
