"""
주행 Q&A 브리지 — 인식 결과 → LLM 컨텍스트 → Claude Haiku 스트리밍 답변

재활용: Anthropic Python SDK (anthropic>=0.40.0)

사용법:
  # .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 설정 후:
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

    lines = [f"현재 인식된 객체 (총 {len(tracks)}개):"]
    for t in tracks:
        cls_name = "교통표지판" if t["class"] == 0 else "간판"
        label = t.get("label") or t.get("class_name", "")
        conf_pct = int(t["conf"] * 100)

        # label이 없거나 'traffic_sign' 기본값인 경우
        if label and label not in ("traffic_sign", "signboard"):
            label_str = f"'{label}'"
        else:
            label_str = f"(미인식)"

        lines.append(f"  [Track #{t['id']}] {cls_name} - {label_str} (신뢰도 {conf_pct}%)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Claude API 스트리밍 Q&A
# ─────────────────────────────────────────────────────────────────────────────

async def ask_stream(
    context: str,
    question: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 300,
) -> AsyncIterator[str]:
    """
    인식 컨텍스트 + 사용자 질문 → Claude 스트리밍 답변 토큰 이터레이터.

    Args:
        context:   build_context() 반환 문자열
        question:  사용자 질문
        model:     Anthropic 모델 ID
        max_tokens: 최대 응답 토큰 수

    Yields:
        str: 스트리밍 텍스트 토큰
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic 패키지가 설치되지 않았습니다.\n"
            "설치: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        yield "⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)

    user_message = f"{context}\n\n운전자 질문: {question}"

    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
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
    """더미 컨텍스트로 Claude API 연결 테스트."""
    dummy_tracks = [
        {"id": 1, "class": 0, "class_name": "traffic_sign", "conf": 0.92,
         "label": "traffic_sign", "bbox": [100, 100, 200, 200]},
        {"id": 2, "class": 1, "class_name": "signboard",    "conf": 0.85,
         "label": "카페",       "bbox": [300, 150, 500, 280]},
    ]
    context = build_context(dummy_tracks)
    print("=== 컨텍스트 ===")
    print(context)
    print("\n=== Claude 답변 (스트리밍) ===")

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
