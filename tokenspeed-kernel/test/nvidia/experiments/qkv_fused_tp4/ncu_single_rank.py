# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Profile rank zero while uninstrumented peers run the same graph normally."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
script = root / "capture_graph.py"
output = Path(sys.argv[sys.argv.index("--output") + 1])
workers = []
logs = []
common = os.environ.copy()
common.update(
    WORLD_SIZE="4",
    MASTER_ADDR="127.0.0.1",
    MASTER_PORT="29673",
    TORCHELASTIC_USE_AGENT_STORE="False",
)
for rank in range(1, 4):
    env = common.copy()
    env.update(RANK=str(rank), LOCAL_RANK=str(rank))
    # ncu --target-processes application-only does not instrument children.
    for name in list(env):
        if name.startswith("NV_COMPUTE_PROFILER") or name == "CUDA_INJECTION64_PATH":
            del env[name]
    log = output.with_name(f"{output.stem}-peer-rank{rank}.log").open("w")
    logs.append(log)
    workers.append(
        subprocess.Popen(
            [sys.executable, str(script), *sys.argv[1:]],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    )
os.environ.update(common)
os.environ.update(RANK="0", LOCAL_RANK="0")
try:
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
    codes = [worker.wait() for worker in workers]
    assert not any(codes), codes
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    for worker in workers:
        worker.wait()
    for log in logs:
        log.close()
