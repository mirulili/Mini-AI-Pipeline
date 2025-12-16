# src/generator.py
from __future__ import annotations

import re
from openai import OpenAI
from . import config, utils

SYSTEM_MSG = (
    "너는 신중한 한국어 법률 도우미야. 아래 '검색 문맥'에 있는 정보만 근거로 사용해. "
    "먼저 정답을 도출하는 추론 과정(Reasoning)을 작성하고, "
    "마지막 줄에 '정답: X' (X는 A, B, C, D 중 하나) 형식으로 답을 출력해."
)

RULES = (
    "규칙:\n"
    "1) 반드시 '검색 문맥'의 사실만을 근거로 사용한다.\n"
    "2) '검색 문맥'의 사실을 바탕으로 하여 분석 및 추론할 수 있다.\n"
    "3) 단, '검색 문맥'의 정답(정답 라벨)은 현재 문제의 정답(정답 라벨)과 별개이다.\n"
    "4) 가장 타당한 보기를 1개 고른다.\n"
    "5) 단계별로 추론(Chain of Thought)을 서술한다.\n"
    "6) 추론이 끝난 후 마지막 줄에 반드시 '정답: [A/B/C/D]' 형식으로만 결론을 낸다.\n"
)

FEW_SHOT_EXAMPLE = (
    "Example (Few-Shot):\n"
    "[Search Context]\n"
    "질문: 살인죄에 대한 설명으로 옳은 것은?\n"
    "선택지:\n"
    "A) 과실치사는 살인죄에 포함된다.\n"
    "B) 미필적 고의에 의한 살인은 인정되지 않는다.\n"
    "C) 사람을 살해하면 5년 이상의 징역에 처할 수 있다.\n"
    "D) 살인의 고의는 반드시 확정적이어야 한다.\n"
    "정답: C\n"
    "\n"
    "[Question]\n"
    "살인죄에 대한 설명으로 옳은 것은?\n"
    "A) 살인의 고의는 반드시 확정적이어야 한다.\n"
    "B) 사람을 살해하면 5년 이상의 징역에 처할 수 있다.\n"
    "C) 미필적 고의에 의한 살인은 인정되지 않는다.\n"
    "D) 과실치사는 살인죄에 포함된다.\n"
    "\n"
    "Output:\n"
    "'검색 문맥'에서, 본 문제와 유사한 사례를 찾아, 본 문제 정답의 근거로 삼습니다.\n"
    "문맥에 따르면, 정답 C(\"사람을 살해하면 5년 이상의 징역\")가 올바른 정답으로 제시되었습니다.\n"
    "문맥의 정답 C는 본 문제의 선택지 B와 유사합니다."
    "또한, 문맥의 틀린 선택지(A, B, D)를 고려하면, 본 문제의 A, C, D는 틀린 선택지입니다.\n"
    "따라서 문맥에 부합하는 가장 적절한 답은 B입니다.\n"
    "정답: B\n"
)

def parse_answer(text: str | None) -> str:
    """Extract A/B/C/D from LLM output."""
    text = (text or "").strip()
    # Look for '정답: X' pattern near the last line
    # Or find the first alphabet after '정답:' in the whole text
    # Safely search the latter part
    
    # 1. Find '정답:' keyword
    match = re.search(r"정답\s*[:]\s*([ABCD])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. If not found, find A/B/C/D in the entire text
    last_line = text.splitlines()[-1]
    m = re.search(r"[ABCD]", last_line, re.IGNORECASE)
    if m:
        return m.group(0).upper()
        
    raise ValueError(f"Failed to parse answer from LLM output: '{text}'")

def _ask_llm(client: OpenAI, context: str, question_block: str) -> str:
    prompt = (
        f"{RULES}\n\n"
        f"{FEW_SHOT_EXAMPLE}\n\n"
        f"[Search Context]\n{context}\n\n"
        f"[Question]\n{question_block}\n\n"
        "Output:\n"
    )
    
    # gpt-4o-mini is default (configurable in config.py)
    model_name = getattr(config, "MODEL_GENERATION", "gpt-4o-mini")

    out = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1, 
        max_tokens=1000, 
    )
    # Return full response (including CoT) as is
    return out.choices[0].message.content or ""

def call_llm(question_block: str, context: str) -> str:
    client = utils.get_openai_client()
    answer = _ask_llm(client, context, question_block)
    return answer