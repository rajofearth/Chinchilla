import os
from pathlib import Path

from eval_runner import _health_hosts, _native_arg, _normalize_windows_path, _path


def test_normalize_collapses_yaml_double_backslashes():
    assert _normalize_windows_path(r"C:\\Users\\x") == r"C:\Users\x"


def test_path_converts_windows_drive_on_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    p = _path(r"C:\Users\Yashraj\model.gguf")
    assert str(p).replace("\\", "/") == "/mnt/c/Users/Yashraj/model.gguf"


def test_native_arg_rewrites_mnt_for_windows_exe(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    exe = Path("/mnt/p/Applications/llamacpp/llama-server.exe")
    model = Path("/mnt/c/Users/Yashraj/.cache/model.gguf")
    assert _native_arg(model, exe) == r"C:\Users\Yashraj\.cache\model.gguf"
    assert str(exe) == "/mnt/p/Applications/llamacpp/llama-server.exe"


def test_native_arg_keeps_posix_for_linux_binary(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    exe = Path("/usr/bin/llama-server")
    model = Path("/mnt/c/Users/Yashraj/.cache/model.gguf")
    assert _native_arg(model, exe) == str(model)


def test_health_hosts_include_localhost():
    assert "127.0.0.1" in _health_hosts()
