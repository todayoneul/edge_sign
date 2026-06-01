"""Q&A BYOK(방문자 본인 Groq 키) 계약 테스트.

공개 HF Space에서는 서버 키 대신 요청별 api_key를 받는다.
네트워크 호출 없이 '키 없음' 분기만 결정적으로 검증.
"""
import asyncio

from src.pipeline.qa_bridge import ask_stream


def _collect(**kwargs):
    async def run():
        return [tok async for tok in ask_stream("컨텍스트", "질문?", **kwargs)]
    return asyncio.run(run())


def test_no_key_anywhere_yields_guidance(monkeypatch):
    """인자도 없고 env도 없으면 안내 메시지만 yield하고 네트워크 호출 안 함."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = "".join(_collect(api_key=None))
    assert "GROQ_API_KEY" in out or "키" in out


def test_explicit_key_arg_takes_precedence(monkeypatch):
    """api_key 인자가 env보다 우선이며 클라이언트에 그대로 전달된다(네트워크 없이 검증)."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import groq

    class FakeClient:
        def __init__(self, api_key=None):
            assert api_key == "gsk_byok"          # 인자 키가 전달됨
            self.chat = self
            self.completions = self

        async def create(self, **kw):
            raise RuntimeError("SENTINEL_USED_KEY")   # create까지 도달 = 키 사용됨

    monkeypatch.setattr(groq, "AsyncGroq", FakeClient)
    out = "".join(_collect(api_key="gsk_byok"))
    assert "GROQ_API_KEY가 설정되지 않았습니다" not in out
    assert "SENTINEL_USED_KEY" in out
