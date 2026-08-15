import pytest
from eval_runner import score

def test_score_contains():
    assert score({"expect": {"contains": "Paris"}}, "The capital is Paris.")["pass"] is True
    assert score({"expect": {"contains": "Paris"}}, "London")["pass"] is False

def test_score_tool():
    text = 'get_weather(location="Boston")'
    r = score({"expect": {"tool": "get_weather", "args": ["location"]}}, text)
    assert r["pass"] is True
