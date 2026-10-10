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

"""Add the same nine phase markers to a scheduling candidate."""

import argparse
import ast
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
text = args.source.read_text()
original = text


def insert_before(needle, marker):
    global text
    assert text.count(needle) == 1, needle
    indent = needle.splitlines()[0][
        : len(needle.splitlines()[0]) - len(needle.splitlines()[0].lstrip())
    ]
    text = text.replace(needle, f'{indent}iket.mark("{marker}")\n' + needle, 1)


def insert_at_statement(statement, before, after):
    """Keep markers outside complete statements, including wrapped calls."""
    global text
    lines = text.splitlines(keepends=True)
    line = lines[statement.lineno - 1]
    indent = line[: len(line) - len(line.lstrip())]
    if after is not None:
        lines.insert(statement.end_lineno, f'{indent}iket.mark("{after}")\n')
    if before is not None:
        lines.insert(statement.lineno - 1, f'{indent}iket.mark("{before}")\n')
    text = "".join(lines)


def insert_at_call(name, before, after):
    matches = []
    for statement in ast.walk(ast.parse(text)):
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        function = statement.value.func
        if isinstance(function, ast.Name) and function.id == name:
            matches.append(statement)
        elif isinstance(function, ast.Attribute) and function.attr == name:
            matches.append(statement)
    assert len(matches) == 1, name
    insert_at_statement(matches[0], before, after)


text = text.replace(
    "import cutlass.cute as cute\n",
    "import cutlass.cute as cute\nfrom cutlass.cute.experimental import iket\n",
    1,
)
insert_at_call("stage_input", "entry", None)
insert_before(
    "        # Publish all data and scale stores from this CTA before its arrival.",
    "input_stores_done",
)
if "    def await_input_owner(" in text:
    insert_before(
        "        # Only the TMA and scale producers read the communicated activation.",
        "local_publication_done",
    )
else:
    insert_before(
        "        for peer in cutlass.range_constexpr(len(peer_flags)):",
        "local_publication_done",
    )
insert_at_call("pipeline_init_wait", None, "roles_start")
insert_before(
    "        # All roles have retired. Reuse the whole resident grid for the",
    "math_role_done",
)
insert_at_call("phase_exchange", "exchange_start", "exit")
assignments = [
    node
    for node in ast.walk(ast.parse(text))
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "tile_count"
        for target in node.targets
    )
]
assert len(assignments) == 1
insert_at_statement(assignments[0], "output_ready", None)
insert_before(
    "        # CTA zero acquires each per-CTA arrival in parallel and publishes",
    "copy_done",
)
stripped = "".join(
    line
    for line in text.splitlines(keepends=True)
    if "iket.mark(" not in line
    and "from cutlass.cute.experimental import iket" not in line
)
assert stripped == original
args.output.write_text(text)
