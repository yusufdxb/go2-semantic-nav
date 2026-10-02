import os

from go2_open_vocab_detector.backends.ultralytics_detector import resolve_weight_file


def test_prefers_cache_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache" / "ultralytics"
    cache.mkdir(parents=True)
    (cache / "w.pt").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert resolve_weight_file("w.pt") == os.path.join(str(cache), "w.pt")


def test_falls_back_to_cwd_then_bare_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert resolve_weight_file("w.pt") == "w.pt"
    (tmp_path / "w.pt").write_bytes(b"x")
    assert resolve_weight_file("w.pt") == os.path.join(str(tmp_path), "w.pt")
