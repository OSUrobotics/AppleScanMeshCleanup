"""Legacy seam CSV compatibility for alignment and texture filling."""

import csv
from pathlib import Path

import numpy as np

from filename_fixing import _split_filename_into_apple_type_and_number
from normalize_uv_projection import _find_seam_file


def _read_legacy_seam_data(
    seam_path: Path,
) -> tuple[list[tuple[list[int], np.ndarray]], np.ndarray | None]:
    with seam_path.open(encoding="utf-8", newline="") as seam_file:
        rows = [[entry.strip() for entry in row] for row in csv.reader(seam_file)]
    if len(rows) not in (18, 20):
        raise ValueError(f"Expected 18 or 20 seam CSV rows in {seam_path}, found {len(rows)}.")
    seams = []
    for start in range(0, 18, 3):
        group = [row[:-1] if row and not row[-1] else row for row in rows[start:start + 3]]
        if not group[0] or len({len(row) for row in group}) != 1:
            raise ValueError(f"Seam rows {start + 1}-{start + 3} have inconsistent lengths.")
        ids = [int(value) for value in group[0]]
        if min(ids) < 0:
            raise ValueError(f"Negative mesh vertex index in {seam_path}.")
        points = np.column_stack((
            np.asarray(group[1], dtype=np.float64),
            np.asarray(group[2], dtype=np.float64),
        ))
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite seam coordinates in {seam_path}.")
        seams.append((ids, points))
    centers = None
    if len(rows) == 20:
        try:
            centers = np.asarray([[float(value) for value in row] for row in rows[18:]], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Expected two finite 3D center coordinates in {seam_path}.") from exc
        if centers.shape != (2, 3) or not np.isfinite(centers).all():
            raise ValueError(f"Expected two finite 3D center coordinates in {seam_path}.")
    return seams, centers


def _find_compatible_seam_file(input_path: Path, seam_dir: Path, *, legacy_only: bool = False) -> Path:
    if not seam_dir.is_dir():
        raise NotADirectoryError(f"Seam directory not found or not a directory: {seam_dir}")
    if not legacy_only:
        try:
            return _find_seam_file(input_path, seam_dir)
        except FileNotFoundError:
            pass
    apple = _split_filename_into_apple_type_and_number(input_path.name)
    matches = []
    for candidate in seam_dir.glob("*_seam.csv"):
        if not candidate.is_file():
            continue
        try:
            candidate_apple = _split_filename_into_apple_type_and_number(candidate.name)
        except ValueError:
            continue
        if candidate_apple == apple:
            matches.append(candidate)
    if len(matches) > 1:
        names = ", ".join(sorted(path.name for path in matches))
        raise ValueError(f"Multiple seam CSV files match {input_path.name} in {seam_dir}: {names}")
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No seam CSV found or topology JSON found for {input_path.name} in {seam_dir}.")
