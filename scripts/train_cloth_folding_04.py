"""Run official normalization, then train with an automatic batch-size fallback."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs/cloth_folding_04"
CONFIG = "pi05_cloth_folding_04"
EXPERIMENT = "black_cloth_50k"
PYTHON = ROOT / ".venv/bin/python"


def status(phase, **details):
    record = {"phase": phase, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **details}
    (LOGS / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


def run(command, log, env):
    with log.open("w") as output:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        print(f"Child PID {child.pid}: {' '.join(map(str, command))}", flush=True)
        return child.wait()


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_LEROBOT_HOME=str(ROOT), HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    status("normalization")
    result = run(
        [str(PYTHON), "scripts/compute_norm_stats.py", "--config-name", CONFIG],
        LOGS / "norm_stats.log",
        dict(env, JAX_PLATFORMS="cpu"),
    )
    if result:
        status("failed", stage="normalization", returncode=result)
        return result

    env.pop("JAX_PLATFORMS", None)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for batch_size in (4, 2):
        status("training", batch_size=batch_size)
        log = LOGS / f"train_bs{batch_size}.log"
        command = [
            str(PYTHON), "scripts/train.py", CONFIG,
            "--exp-name", EXPERIMENT, "--batch-size", str(batch_size),
        ]
        if (ROOT / "checkpoints" / CONFIG / EXPERIMENT).exists():
            command.append("--resume")
        result = run(command, log, env)
        if result == 0:
            status("complete", batch_size=batch_size)
            return 0
        message = log.read_text(errors="replace").lower()
        oom = any(term in message for term in (
            "out of memory", "resource_exhausted", "cuda_error_out_of_memory",
            "failed to allocate", "cudnn_status_alloc_failed",
        ))
        if batch_size == 4 and oom:
            status("retrying", reason="GPU memory exhausted", batch_size=2)
            time.sleep(5)
            continue
        status("failed", stage="training", batch_size=batch_size, returncode=result)
        return result
    return 1


if __name__ == "__main__":
    sys.exit(main())
