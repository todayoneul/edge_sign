"""
주행 Q&A 브리지 — 인식 결과 → LLM 컨텍스트 → Groq LLM 스트리밍 답변

재활용: Groq Python SDK (groq>=0.11.0). OpenAI 호환 Chat Completions 인터페이스.

사용법:
  # .env 파일에 GROQ_API_KEY=gsk_... 설정 후:
  python src/pipeline/qa_bridge.py --test

  # 코드에서 직접 사용:
  from src.pipeline.qa_bridge import build_context, ask_stream
  context = build_context(result["tracks"])
  async for token in ask_stream(context, "저 표지판이 뭐야?"):
      print(token, end="", flush=True)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import AsyncIterator

# .env 로드 (python-dotenv)
def _load_dotenv():
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv 없으면 환경변수 직접 설정

_load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 주행 중인 운전자를 돕는 지능형 어시스턴트입니다.
차량에 부착된 카메라가 실시간으로 교통표지판과 한글 간판을 인식하고 있습니다.
인식된 객체 목록이 컨텍스트로 제공됩니다.

답변 규칙:
1. 인식된 객체와 관련된 질문에 우선 답변하세요.
2. 운전 안전과 직결된 정보(속도제한, 정지 신호 등)는 먼저 언급하세요.
3. 간결하게 1~3문장으로 답변하세요 (운전 중 집중을 고려).
4. 인식 결과가 불확실한 경우 그 점을 솔직히 알려주세요.
5. 항상 한국어로 답변하세요."""


# 기본 모델 — Groq 프로덕션 모델 중 한국어 품질이 좋은 Llama 3.3 70B.
# 더 빠른 응답이 필요하면 "llama-3.1-8b-instant" 로 교체 가능.
DEFAULT_MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────────────────────
# 컨텍스트 빌더
# ─────────────────────────────────────────────────────────────────────────────

def build_context(tracks: list[dict]) -> str:
    """
    파이프라인 결과(tracks) → 자연어 컨텍스트 문자열.

    Args:
        tracks: e2e_pipeline.process_frame()["tracks"] 리스트

    Returns:
        예: "현재 인식된 객체 (총 2개):
              [Track #2] 교통표지판 - 'traffic_sign' (신뢰도 94%)
              [Track #5] 간판 - '카페' (신뢰도 87%)"
    """
    if not tracks:
        return "현재 카메라에서 인식된 객체가 없습니다."

    _KIND = {0: "교통표지판", 1: "신호등", 2: "간판"}
    lines = [f"현재 인식된 객체 (총 {len(tracks)}개):"]
    for t in tracks:
        cls_name = _KIND.get(t["class"], "객체")
        label = t.get("label") or t.get("class_name", "")
        conf_pct = int(t["conf"] * 100)

        # label이 분류 결과(한국어)면 표시, 검출 기본값(영문 class명)이면 미인식
        if label and label not in ("traffic_sign", "traffic_light", "signboard"):
            label_str = f"'{label}'"
        else:
            label_str = "(세부 미인식)"

        lines.append(f"  [Track #{t['id']}] {cls_name} - {label_str} (신뢰도 {conf_pct}%)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Groq API 스트리밍 Q&A
# ─────────────────────────────────────────────────────────────────────────────

async def ask_stream(
    context: str,
    question: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
) -> AsyncIterator[str]:
    """
    인식 컨텍스트 + 사용자 질문 → Groq 스트리밍 답변 토큰 이터레이터.

    Args:
        context:   build_context() 반환 문자열
        question:  사용자 질문
        model:     Groq 모델 ID (기본: llama-3.3-70b-versatile)
        max_tokens: 최대 응답 토큰 수

    Yields:
        str: 스트리밍 텍스트 토큰
    """
    try:
        from groq import AsyncGroq
    except ImportError:
        raise ImportError(
            "groq 패키지가 설치되지 않았습니다.\n"
            "설치: pip install groq"
        )

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        yield "⚠️ GROQ_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
        return

    client = AsyncGroq(api_key=api_key)

    user_message = f"{context}\n\n운전자 질문: {question}"

    try:
        stream = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:
        yield f"\n⚠️ API 오류: {e}"


async def ask_once(context: str, question: str, **kwargs) -> str:
    """ask_stream을 완전히 소비하여 전체 답변 문자열 반환 (테스트용)."""
    parts = []
    async for token in ask_stream(context, question, **kwargs):
        parts.append(token)
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 테스트
# ─────────────────────────────────────────────────────────────────────────────

async def _test():
    """더미 컨텍스트로 Groq API 연결 테스트."""
    dummy_tracks = [
        {"id": 1, "class": 0, "class_name": "traffic_sign",  "conf": 0.92,
         "label": "속도제한50",   "bbox": [100, 100, 200, 200]},
        {"id": 2, "class": 1, "class_name": "traffic_light", "conf": 0.85,
         "label": "신호등_빨강",  "bbox": [300, 150, 360, 260]},
    ]
    context = build_context(dummy_tracks)
    print("=== 컨텍스트 ===")
    print(context)
    print("\n=== Groq 답변 (스트리밍) ===")

    question = "지금 보이는 것들에 대해 간단히 설명해줘."
    async for token in ask_stream(context, question):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        asyncio.run(_test())
