import json
import os
import tempfile
from types import SimpleNamespace
from scripts.core.callbacks import DiagnosticCallback


def _fake_state(step):
    return SimpleNamespace(global_step=step)


def _fake_args():
    return SimpleNamespace()


def _fake_control():
    return SimpleNamespace()


def test_callback_writes_jsonl_records():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    cb = DiagnosticCallback(output_path=path)
    cb.on_train_begin(_fake_args(), _fake_state(0), _fake_control())
    cb.on_log(_fake_args(), _fake_state(10), _fake_control(),
              logs={"loss": 0.5, "learning_rate": 0.0002})
    cb.on_log(_fake_args(), _fake_state(20), _fake_control(),
              logs={"loss": 0.4, "learning_rate": 0.0001})
    cb.on_train_end(_fake_args(), _fake_state(20), _fake_control())

    with open(path) as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) == 2
    assert lines[0]["step"] == 10
    assert lines[0]["loss"] == 0.5
    assert lines[1]["step"] == 20
    os.unlink(path)


def test_callback_handles_none_logs():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    cb = DiagnosticCallback(output_path=path)
    cb.on_train_begin(_fake_args(), _fake_state(0), _fake_control())
    cb.on_log(_fake_args(), _fake_state(5), _fake_control(), logs=None)
    cb.on_train_end(_fake_args(), _fake_state(5), _fake_control())
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 0
    os.unlink(path)
