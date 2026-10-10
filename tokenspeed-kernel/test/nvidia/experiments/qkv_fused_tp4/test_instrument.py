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

"""CPU regression checks for instrumentation of formatter-wrapped calls."""

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class InstrumentationTest(unittest.TestCase):
    def test_formatted_kernels_keep_source_and_nine_phase_markers(self):
        root = Path(__file__).resolve().parent
        kernels = (
            root.parents[3]
            / "python/tokenspeed_kernel/thirdparty/cute_dsl/fused_tp4_projection"
        )
        for variant in ("baseline", "optimized"):
            with self.subTest(
                variant=variant
            ), tempfile.TemporaryDirectory() as directory:
                source = kernels / f"{variant}_kernel.py"
                output = Path(directory) / "instrumented.py"
                subprocess.run(
                    [
                        sys.executable,
                        str(root / "instrument.py"),
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
                instrumented = output.read_text()
                tree = ast.parse(instrumented)
                markers = [
                    node.value.args[0].value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "iket"
                    and node.value.func.attr == "mark"
                ]
                self.assertCountEqual(
                    markers,
                    (
                        "entry",
                        "input_stores_done",
                        "local_publication_done",
                        "roles_start",
                        "math_role_done",
                        "exchange_start",
                        "output_ready",
                        "copy_done",
                        "exit",
                    ),
                )
                stripped = "".join(
                    line
                    for line in instrumented.splitlines(keepends=True)
                    if "iket.mark(" not in line
                    and "from cutlass.cute.experimental import iket" not in line
                )
                self.assertEqual(stripped, source.read_text())


if __name__ == "__main__":
    unittest.main()
