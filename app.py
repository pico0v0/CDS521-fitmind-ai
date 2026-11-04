"""运行：streamlit run app.py。首版实现规则 Demo + 可选大模型编辑意图解析。"""
import os
import streamlit as st
from src.core import (DAYS, read_catalog, build_demo_plan, validate_plan,
                      parse_edit_locally, apply_edit)
from src.llm_client import parse_edit_with_llm

st.set_page_config(page_title="FitMind AI — 课程项目 Demo", page_icon="💪", layout="wide")
st.title("FitMind AI｜个性化健身与饮食规划")
st.caption("教学演示版。基础计划采用规则组合；勾选 API 选项后，调整指令由真实 LLM 解析。非医疗建议。")
with st.sidebar:
    st.header("用户画像")
    age = st.number_input("年龄", 18, 64, 23)
    height = st.number_input("身高 cm", 130, 220, 170)
    weight = st.number_input("体重 kg", 30, 250, 65)
    goal = st.selectbox("目标", ["减脂", "增肌", "塑形"])
    experience = st.selectbox("训练经验", ["初学者", "有经验"])
    equipment = st.selectbox("器械", ["徒手", "哑铃", "健身房"])
    available_days = st.multiselect("可训练日", DAYS, default=["周一","周三","周四","周五"])
    sessions = st.number_input("每周训练次数", 1, 6, 3)
    allergies = st.multiselect("过敏类别", ["乳制品","鸡蛋","小麦","大豆","鱼类","坚果"])
    _, foods, _ = read_catalog()
    avoid_names = st.multiselect("不吃的食物", [f["name"] for f in foods])
    ai_intent = st.checkbox("使用大模型解析调整指令（需要 API Key）", value=False)
    btn = st.button("生成 7 天示例计划", type="primary", use_container_width=True)
if btn:
    profile = {"age":int(age),"height_cm":int(height),"weight_kg":float(weight),
               "goal":goal,"experience":experience,"equipment":equipment,
               "available_days":available_days,"sessions":int(sessions),"allergies":allergies,
               "exclude_foods":[f["id"] for f in foods if f["name"] in avoid_names]}
    try:
        st.session_state.plan = build_demo_plan(profile)
    except ValueError as e:
        st.error(str(e))
if "plan" in st.session_state:
    plan = st.session_state.plan
    issues = validate_plan(plan)
    st.success("现有规则检查通过" if not issues else "规则检查发现问题")
    if issues:
        st.error("；".join(issues))
    tabs = st.tabs(DAYS)
    for tab, day in zip(tabs, plan["week"]):
        with tab:
            w = day["workout"]
            st.markdown("**训练：** " + (", ".join(w["exercise_ids"]) + f"（约 {w['minutes']} 分钟）" if w else "恢复/休息日"))
            for slot, meal in day["meals"].items():
                st.write(f"**{slot}：** {meal['name']} · {meal['nutrition']['kcal']:.0f} kcal（数据库示例估算）")
    st.divider()
    text = st.text_input("自然语言调整（例如：周三挪到周四，不要鸡蛋）", value="周三挪到周四，不要鸡蛋")
    if st.button("调整计划"):
        try:
            if ai_intent:
                allowed = [{"id":f["id"], "name":f["name"]} for f in foods]
                intent = parse_edit_with_llm(text, allowed)
                intent["exclude_foods"] = [x for x in intent["exclude_foods"] if x in {f["id"] for f in foods}]
            else:
                intent = parse_edit_locally(text, foods)
            st.session_state.plan = apply_edit(plan, intent)
            st.success("修改成功，已复检。请检查新计划。")
            st.json(intent)
        except Exception as e:
            st.error(f"未完成调整：{e}")
    st.download_button("下载结构化计划 JSON", __import__('json').dumps(plan,ensure_ascii=False,indent=2), file_name="plan.json")
st.info("健康边界：仅适用于健康成年人的课堂 Demo；不处理受伤康复、疾病、孕期等医学场景。营养份量尚未针对个人能量需求优化。")
