import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import dashboard.app as dash


@pytest.mark.anyio
async def test_post_run_streams_events_and_persists(tmp_path, monkeypatch):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    monkeypatch.setattr(dash, "RESULTS", out_dir)

    def fake_run(models_path, cases_path, selected, output, on_event, run_id=None):
        assert selected == ["base"]
        on_event({"type": "prompt", "model": "base", "case": "qa.math", "prompt": "1+1"})
        on_event({"type": "finished", "model": "base", "case": "qa.math", "response": "2", "score": {"pass": True}, "metrics": {"elapsed_ms": 10}})
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "models": [{"id": "base", "label": "Base", "responses": [{"case_id": "qa.math", "text": "2", "score": {"pass": True}, "elapsed_ms": 10}]}],
            "summary": {"base": {"passed": 1, "completed": 1, "total": 1}},
        }
        Path(output).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(dash, "run", fake_run)
    transport = ASGITransport(app=dash.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/runs", json={"models": ["base"]})
        assert res.status_code == 200, res.text
        body = res.json()
        text = ""
        async with client.stream("GET", body["events_url"]) as stream:
            async for chunk in stream.aiter_text():
                text += chunk
                if "run_finished" in text:
                    break
        assert "prompt" in text
        assert "finished" in text
        saved = await client.get(f"/api/runs/{body['run_id']}")
        assert saved.status_code == 200
        data = saved.json()
        assert data["run_id"] == body["run_id"]
        assert data["models"][0]["responses"][0]["case_id"] == "qa.math"


def test_post_run_rejects_unknown_model():
    from fastapi.testclient import TestClient
    client = TestClient(dash.app)
    res = client.post("/api/runs", json={"models": ["does-not-exist"]})
    assert res.status_code == 400
