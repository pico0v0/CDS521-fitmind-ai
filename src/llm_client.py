"""FitMind AI 大模型客户端封装
支持标准 OpenAI 协议（兼容 agentrouter.org / 官方 OpenAI / DeepSeek 等），
通过 Responses API 与结构化 JSON Schema 保证模型输出符合 Pydantic 契约。
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Type, TypeVar, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

# 优先读取 fitmind-ai/.env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://agentrouter.org/v1")
DEFAULT_MODEL = os.getenv("FITMIND_MODEL", "gpt-5.6-sol")


def get_openai_client() -> OpenAI:
    """初始化 OpenAI 兼容客户端，包含防拦截 User-Agent 标头"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量，请在 fitmind-ai/.env 中配置。")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={"User-Agent": "Cline/3.0.0"}
    )


def call_structured_llm(
    response_model: Type[T],
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_retries: int = 2,
    timeout: float = 240.0,
) -> tuple[T, dict[str, Any]]:
    """
    通过 Responses API 强类型解析输出符合 response_model 的结构化数据。
    返回 (parsed_result, meta_info: {latency_ms, tokens_in, tokens_out, model})
    """
    client = get_openai_client()
    target_model = model or os.getenv("FITMIND_MODEL", DEFAULT_MODEL)
    
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            response = client.responses.parse(
                model=target_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                text_format=response_model,
                timeout=timeout
            )
            latency_ms = int((time.time() - t0) * 1000)
            
            if response.output_parsed is None:
                raise RuntimeError(f"模型未返回结构化内容: {response}")
            
            usage = getattr(response, "usage", None)
            tokens_in = getattr(usage, "input_tokens", 0) if usage else 0
            tokens_out = getattr(usage, "output_tokens", 0) if usage else 0
            
            meta = {
                "latency_ms": latency_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": target_model,
                "attempt": attempt,
            }
            return response.output_parsed, meta
            
        except Exception as e:
            last_error = e
            time.sleep(1.0 * attempt)
            
    raise RuntimeError(f"LLM 结构化调用失败（重试 {max_retries} 次）：{last_error}")


def parse_edit_with_llm(text: str, allowed_foods: list[dict], model: Optional[str] = None) -> dict[str, Any]:
    """自然语言编辑意图抽取"""
    from src.schemas import EditIntent
    
    sys_prompt = (
        "你是健身与饮食规划系统的意图解析专家，职责是从用户的自然语言反馈中精确提取出结构化修改指令。\n"
        "要求：\n"
        "1. 日期格式必须严格为周一至周日（如'周三'、'周四'）。\n"
        "2. 如果用户提到不想吃某种食物，必须匹配并从允许的候选列表中提取其 food_id，严禁自行编造。\n"
        "3. 未明确提及的字段填写 null 或空列表。\n"
        f"允许的食品库 ID 参照：{allowed_foods}"
    )
    user_prompt = f"用户修改指令：{text}"
    
    parsed, _ = call_structured_llm(
        response_model=EditIntent,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        model=model
    )
    return parsed.model_dump()
