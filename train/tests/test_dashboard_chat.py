import asyncio
import json
from pathlib import Path

import dashboard.app as dashboard


def test_chat_persists_selected_model(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "CHATS", tmp_path)
    chat = dashboard.create_chat(dashboard.ChatCreate(model="grpo-5500", title="checkpoint test"))

    saved = json.loads((tmp_path / f"{chat['id']}.json").read_text(encoding="utf-8"))
    assert saved["model"] == "grpo-5500"
    assert dashboard.read_chat(chat["id"])["model"] == "grpo-5500"
    assert dashboard.list_chats()[0]["model"] == "grpo-5500"


def test_chat_stream_delivers_reply_and_saves_reasoning(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "CHATS", tmp_path)
    monkeypatch.setattr(
        dashboard,
        "load_matrix",
        lambda _: ({"server": {"port": 18080, "temperature": 0.2, "max_tokens": 32}}, [{"id": "base", "label": "Base", "lora": None}]),
    )

    class FakeServer:
        port = 18080
        client_host = "127.0.0.1"

    class FakeClient:
        def __init__(self, port, host):
            assert (port, host) == (18080, "127.0.0.1")

        def stream(self, model, messages, generation, on_delta=None, on_chunk=None):
            assert model == "base"
            assert messages[-1] == {"role": "user", "content": "hello"}
            on_chunk({"reasoning": "thinking", "text": ""})
            on_chunk({"reasoning": "", "text": "hello back"})
            return {
                "elapsed_ms": 12,
                "ttft_ms": 2,
                "generation_ms": 10,
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "tok_per_sec": 200,
                "token_source": "server_usage",
            }

    monkeypatch.setattr(dashboard, "ensure_chat_server", lambda cfg, model: FakeServer())
    monkeypatch.setattr(dashboard, "OpenAIClient", FakeClient)
    dashboard.chat_server_lock = asyncio.Lock()

    chat = dashboard.create_chat(dashboard.ChatCreate(model="base"))

    async def collect():
        response = await dashboard.send_chat_message(chat["id"], dashboard.ChatMessage(content="hello"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return "".join(chunks)

    body = asyncio.run(collect())
    assert "data: {" in body
    assert "\n\n" in body
    assert '"type": "reasoning"' in body
    assert '"type": "delta"' in body
    assert '"type": "done"' in body
    assert body.rstrip().endswith('"type": "close"}')

    saved = dashboard.read_chat(chat["id"])
    assistant = saved["messages"][-1]
    assert assistant["content"] == "hello back"
    assert assistant["reasoning"] == "thinking"
    assert assistant["status"] == "completed"
    assert assistant["completion_tokens"] == 2
