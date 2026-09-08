"""Export the full prescribed endpoint record from one explicit evaluation.

This formats existing measurements; it makes no prediction or success decision.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evaluation_path = args.evaluation.resolve()
    raw = evaluation_path.read_bytes()
    evaluation = json.loads(raw)
    rows = evaluation["future"]
    lookup = {(r["time_fs"], r["path_index"]): r for r in rows}
    times = sorted({r["time_fs"] for r in rows})
    if len(rows) != 40 or len(lookup) != 40 or len(times) != 10:
        raise ValueError("Expected the complete 40-row prescribed record.")
    if set(lookup) != {(t, p) for t in times for p in range(4)}:
        raise ValueError("Endpoint identities do not form four complete paths.")
    counts = Counter(r["status"] for r in rows)
    if set(counts) - {"complete", "pending", "failed"}:
        raise ValueError(f"Unrecognized reference status: {counts}")
    horizons = {
        p["path_index"]: p["frozen_horizons_fs"]["0.25"] for p in evaluation["paths"]
    }
    source = root / "manuscript/figures/source_data/interface_endpoint_record"
    source.mkdir(parents=True, exist_ok=True)
    columns = [
        "case",
        "path_index",
        "time_fs",
        "status",
        "inside_primary_window",
        "max_atom_error_ev_a",
        "endpoint_work_ev",
    ]
    with (source / "endpoints.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["time_fs"], r["path_index"])):
            record = {k: row.get(k, "") for k in columns}
            record["inside_primary_window"] = (
                row["time_fs"] <= horizons[row["path_index"]]
            )
            writer.writerow(record)

    def cell(row, field, scale, digits):
        if row["status"] != "complete":
            return {"pending": "P", "failed": "F"}[row["status"]]
        value = f"{row[field] * scale:.{digits}f}"
        if (
            field == "max_atom_error_ev_a"
            and row["time_fs"] <= horizons[row["path_index"]]
        ):
            value += r"$^{*}$"
        return value

    def panel(title, field, scale, digits):
        result = [
            r"\begin{minipage}[t]{0.49\textwidth}\centering",
            title + r"\par\smallskip",
            r"\begin{tabular}{rrrrr}\toprule",
            r"$t$ (fs) & p0 & p1 & p2 & p3\\\midrule",
        ]
        for t in times:
            values = [cell(lookup[t, p], field, scale, digits) for p in range(4)]
            result.append(f"{t:g} & " + " & ".join(values) + r"\\")
        result.extend([r"\bottomrule\end{tabular}", r"\end{minipage}"])
        return "\n".join(result)

    caption = (
        r"\caption{\label{tab:interface-endpoints}"
        r"\textbf{Complete prescribed endpoint record for the interface.} "
        r"Paths p0/p1 start from origin 36 and p2/p3 from origin 161. "
        r"Stars mark evaluation times inside the frozen 0.25-eV/\AA\ admission windows. "
        r"Residual work uses the endpoint energy expression. "
        r"The 1.5--4-fs points are continuation probes beyond the declared "
        r"1-fs envelope domain. "
    )
    if counts["pending"]:
        caption += r"P denotes a pending reference evaluation. "
    if counts["failed"]:
        caption += r"F denotes a failed reference attempt. "
    caption += "}"
    tex = [
        "% Generated from "
        + evaluation_path.name
        + "; do not transcribe measurements.",
        r"\subsection{Complete prescribed endpoint record}",
        r"\label{app:interface-record}",
        (
            r"Table~\ref{tab:interface-endpoints} reports the four fixed paths at every prescribed reference time. "
            r"The same initial-force correction defines all residuals and endpoint work on each path."
        ),
        r"\begin{table*}[tp]",
        caption,
        r"\centering\small\setlength{\tabcolsep}{3pt}",
        panel(r"Maximum atomic residual $e$ (eV/\AA)", "max_atom_error_ev_a", 1, 4),
        r"\hfill",
        panel(r"Residual work $W$ (meV)", "endpoint_work_ev", 1000, 3),
        r"\end{table*}",
        "",
    ]
    tex_path = root / "manuscript/interface_prospective_full_record.tex"
    tex_path.write_text("\n".join(tex))
    metadata = {
        "evaluation": str(evaluation_path.relative_to(root)),
        "evaluation_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statuses": dict(counts),
        "time_coordinates_fs": times,
        "primary_windows_fs": horizons,
        "interpretation": "Full prescribed record; formatting only, no scientific decision.",
        "outputs_sha256": {
            "endpoints.csv": hashlib.sha256(
                (source / "endpoints.csv").read_bytes()
            ).hexdigest(),
            tex_path.name: hashlib.sha256(tex_path.read_bytes()).hexdigest(),
        },
    }
    (source / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"statuses": dict(counts), "tex": str(tex_path)}))


if __name__ == "__main__":
    main()
