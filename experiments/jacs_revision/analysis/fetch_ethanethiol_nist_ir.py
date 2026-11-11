#!/usr/bin/env python3
"""Download and convert the NIST gas-phase ethanethiol IR spectrum."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np


NIST_JCAMP_URL = (
    "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C75081&Index=0&Type=IR"
)


def parse_jcamp(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse NIST's X++(Y..Y) JCAMP representation."""
    metadata: dict[str, str] = {}
    rows: list[str] = []
    in_data = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("##"):
            key, separator, value = line[2:].partition("=")
            if separator:
                metadata[key.strip()] = value.strip()
            in_data = key.strip() == "XYDATA"
            continue
        if in_data and line:
            rows.append(line)

    delta_x = float(metadata["DELTAX"]) * float(metadata.get("XFACTOR", "1"))
    y_factor = float(metadata.get("YFACTOR", "1"))
    frequency: list[float] = []
    intensity: list[float] = []
    for row in rows:
        values = [float(value) for value in row.split()]
        x_start, y_values = values[0], values[1:]
        frequency.extend(x_start + delta_x * index for index in range(len(y_values)))
        intensity.extend(value * y_factor for value in y_values)

    n_points = int(metadata["NPOINTS"])
    return np.asarray(frequency[:n_points]), np.asarray(intensity[:n_points])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/revision/external/ethanethiol_nist_gas_ir.csv"),
    )
    args = parser.parse_args()

    request = urllib.request.Request(
        NIST_JCAMP_URL,
        headers={"User-Agent": "equiv-dens-jacs-revision/1.0 (scientific data retrieval)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        frequency, intensity = parse_jcamp(response.read().decode("ascii"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.out,
        np.column_stack((frequency, intensity)),
        delimiter=",",
        header="frequency_cm-1,absorbance",
        comments="",
    )
    print(f"wrote {len(frequency)} NIST points to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
