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

"""Rebuild checked per-warp and Nsight summaries from the saved captures.

IKET intervals are elapsed marked spans, not hardware idle-cycle counts.
Deferred barriers can charge waiting to the following marked interval.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

MARKERS = (
    "entry",
    "input_stores_done",
    "local_publication_done",
    "roles_start",
    "math_role_done",
    "exchange_start",
    "output_ready",
    "copy_done",
    "exit",
)
PHASES = (
    "quant_push",
    "input_publish",
    "setup_and_global_wait",
    "gemm_roles",
    "role_join",
    "output_ready_wait",
    "copy",
    "consumption_join",
)
METRICS = (
    *PHASES,
    "setup_and_roles",
    "pre_exchange",
    "total",
    "ready_copy",
    "output_tail",
)
WARP_ROLES = (
    "accumulator",
    "accumulator",
    "accumulator",
    "accumulator",
    "epilogue",
    "epilogue",
    "epilogue",
    "epilogue",
    "mma",
    "tma_load",
    "scale_load",
    "scheduler",
)


def summarize(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "median_us": statistics.median(ordered),
        "p95_us": ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower),
        "max_us": max(ordered),
        "mean_us": statistics.mean(ordered),
    }


def extract_warps(root, variant):
    records = []
    processes = 0
    for path in sorted((root / f"iket-coarse-{variant}").glob("*.trace.json")):
        data = json.loads(path.read_text())
        if not data["graphLaunches"]:
            continue
        processes += 1
        launches = list(data["graphLaunches"].values())
        assert len(launches[-1]) == 3, path
        kernel = launches[-1][-1]
        assert len(kernel["warpLifetimes"]) == 152 * 12, path
        assert len(kernel["markers"]) == 152 * 12 * len(MARKERS), path
        locations = {}
        for marker in kernel["markers"]:
            location = locations.setdefault(marker["locIdx"], {})
            name = data["stringTable"][marker["markerNameIdx"]]
            assert name not in location, (path, marker)
            location[name] = marker["timestamp"]
        assert len(locations) == 152 * 12, path
        identities = set()
        for index, marks in locations.items():
            assert set(marks) == set(MARKERS), (path, index)
            stamps = [marks[name] for name in MARKERS]
            assert stamps == sorted(stamps), (path, index)
            location = data["locationTable"][index]
            assert location["ctaId"][:2] == [0, 0], location
            identity = (location["ctaId"][2], location["warpId"])
            assert identity not in identities, (path, identity)
            identities.add(identity)
            record = {
                "variant": variant,
                "process": path.stem,
                "cta": identity[0],
                "warp": identity[1],
                "role": WARP_ROLES[identity[1]],
                "sm": location["smId"],
            }
            # Subtract integer nanosecond timestamps before converting to us.
            record.update(
                {
                    f"{phase}_us": (stamps[i + 1] - stamps[i]) / 1000
                    for i, phase in enumerate(PHASES)
                }
            )
            record["setup_and_roles_us"] = (
                marks["math_role_done"] - marks["local_publication_done"]
            ) / 1000
            record["pre_exchange_us"] = (
                marks["exchange_start"] - marks["entry"]
            ) / 1000
            record["total_us"] = (marks["exit"] - marks["entry"]) / 1000
            record["ready_copy_us"] = (
                marks["copy_done"] - marks["exchange_start"]
            ) / 1000
            record["output_tail_us"] = (marks["exit"] - marks["exchange_start"]) / 1000
            records.append(record)
        assert identities == {
            (cta, warp) for cta in range(152) for warp in range(12)
        }, path
    assert processes == 4 and len(records) == 7296, (variant, processes, len(records))
    return records


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def extract_ncu(root, filename):
    with (root / filename).open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 3, filename
    units, values = dict(zip(rows[0], rows[1])), dict(zip(rows[0], rows[2]))
    sample_count = float(values["smsp__pcsamp_sample_count"].replace(",", ""))
    assert float(values["smsp__pcsamp_dropped_bytes"]) == 0
    assert float(values["smsp__pcsamp_buffer_overflow"]) == 0
    stalls = {}
    for reason in (
        "barrier",
        "long_scoreboard",
        "membar",
        "not_selected",
        "short_scoreboard",
        "wait",
    ):
        count = int(
            values[f"smsp__pcsamp_warps_issue_stalled_{reason}"].replace(",", "")
        )
        stalls[reason] = {"count": count, "percent": 100 * count / sample_count}
    source = root / filename.replace(".csv", "-source.csv")
    with source.open() as handle:
        source_rows = list(csv.reader(handle))
    instructions = source_rows[2:]
    source_stats = [
        dict(zip(source_rows[1], row)) for row in instructions if len(row) > 2
    ]
    top_stall_pcs = {}
    for reason in stalls:
        metric = f"smsp__pcsamp_warps_issue_stalled_{reason}"
        selected = sorted(
            source_stats,
            key=lambda row: int(row[metric].replace(",", "")),
            reverse=True,
        )
        top_stall_pcs[reason] = [
            {
                "address": row["Address"],
                "sass": row["Source"].strip(),
                "samples": int(row[metric].replace(",", "")),
            }
            for row in selected[:8]
            if int(row[metric].replace(",", "")) > 0
        ]
    opcodes = Counter()
    for row in instructions:
        if len(row) < 2 or not row[1].strip():
            continue
        words = row[1].split()
        opcode = words[1] if words[0].startswith("@") else words[0]
        opcodes[opcode] += 1
    resource_names = (
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_dynamic",
        "launch__shared_mem_per_block_driver",
        "launch__shared_mem_per_block",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
    )
    return {
        "profiled_duration": {
            "value": values["gpu__time_duration.sum"],
            "unit": units["gpu__time_duration.sum"],
        },
        "pc_samples": sample_count,
        "sample_interval_cycles": {
            "value": values["smsp__pcsamp_interval_cycles"],
            "unit": units["smsp__pcsamp_interval_cycles"],
        },
        "stall_samples": stalls,
        "top_stall_pcs": top_stall_pcs,
        "launch_resources": {
            name: {"value": values[name], "unit": units[name]}
            for name in resource_names
        },
        "static_sass_opcodes": dict(sorted(opcodes.items())),
        "static_local_memory_instructions": sum(
            count for name, count in opcodes.items() if name.startswith(("LDL", "STL"))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    all_records, warp_summary, summary = [], [], {}
    for variant in ("baseline", "optimized"):
        records = extract_warps(args.root, variant)
        all_records.extend(records)
        summary[variant] = {
            "warps": len(records),
            "ctas": len(records) // 12,
            "phases": {
                metric: summarize([r[f"{metric}_us"] for r in records])
                for metric in METRICS
            },
        }
        for warp in range(12):
            selected = [r for r in records if r["warp"] == warp]
            for metric in METRICS:
                warp_summary.append(
                    {
                        "variant": variant,
                        "warp": warp,
                        "role": WARP_ROLES[warp],
                        "metric": metric,
                        **summarize([r[f"{metric}_us"] for r in selected]),
                    }
                )
    write_csv(args.root / "iket-all-warps.csv", all_records)
    write_csv(args.root / "iket-by-warp.csv", warp_summary)
    (args.root / "iket-phase-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    ncu = {
        label: extract_ncu(args.root, filename)
        for label, filename in (
            ("baseline", "ncu-baseline.csv"),
            ("optimized", "ncu-optimized.csv"),
        )
    }
    (args.root / "ncu-comparison.json").write_text(json.dumps(ncu, indent=2) + "\n")
    print(
        "Validated 7,296 complete warp timelines per variant and both Nsight captures."
    )
    for variant, result in summary.items():
        print(variant, "output tail:", result["phases"]["output_tail"])


if __name__ == "__main__":
    main()
