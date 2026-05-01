#!/usr/bin/env python3
"""Merge a foreground 3DGS PLY with a background 3DGS PLY into a single PLY."""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


REQUIRED_SCALAR_FIELDS = (
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)
OPTIONAL_NORMAL_FIELDS = ("nx", "ny", "nz")


def read_gs_ply(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load a 3DGS PLY and return its vertex structured array plus the f_rest_* field names."""
    data = PlyData.read(path)
    if "vertex" not in data:
        raise ValueError(f"{path}: missing 'vertex' element")

    vertex = data["vertex"].data
    names = vertex.dtype.names or ()

    for field in REQUIRED_SCALAR_FIELDS:
        if field not in names:
            raise ValueError(f"{path}: missing required field '{field}'")

    rest_fields = sorted(
        (n for n in names if n.startswith("f_rest_")),
        key=lambda n: int(n.split("_")[2]),
    )
    expected = [f"f_rest_{i}" for i in range(len(rest_fields))]
    if rest_fields != expected:
        raise ValueError(
            f"{path}: f_rest_* fields are not contiguous: {rest_fields}"
        )

    return vertex, rest_fields


def build_merged_dtype(
    fg_names: Tuple[str, ...],
    bg_names: Tuple[str, ...],
    rest_count: int,
) -> np.dtype:
    """Build the output structured dtype using the union of fields from both inputs."""
    fields: List[Tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    has_normals = all(n in fg_names for n in OPTIONAL_NORMAL_FIELDS) and \
                  all(n in bg_names for n in OPTIONAL_NORMAL_FIELDS)
    if has_normals:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
    fields += [(f"f_rest_{i}", "<f4") for i in range(rest_count)]
    fields += [
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
    return np.dtype(fields)


def project_to_dtype(src: np.ndarray, dst_dtype: np.dtype) -> np.ndarray:
    """Project src (structured array) into dst_dtype, zero-filling missing fields."""
    out = np.zeros(len(src), dtype=dst_dtype)
    src_names = set(src.dtype.names or ())
    for name in dst_dtype.names:
        if name in src_names:
            out[name] = src[name].astype(dst_dtype[name], copy=False)
    return out


def resolve_output_path(output: str, default_name: str = "merged.gs.ply") -> str:
    """Resolve --output as either a .ply file path or a directory."""
    if output.lower().endswith(".ply"):
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        return output
    os.makedirs(output, exist_ok=True)
    return os.path.join(output, default_name)


def merge_ply(foreground_path: str, background_path: str, output_path: str) -> None:
    print(f"Reading foreground: {foreground_path}")
    fg_vertex, fg_rest = read_gs_ply(foreground_path)
    print(f"  vertices: {len(fg_vertex)}, SH rest fields: {len(fg_rest)}")

    print(f"Reading background: {background_path}")
    bg_vertex, bg_rest = read_gs_ply(background_path)
    print(f"  vertices: {len(bg_vertex)}, SH rest fields: {len(bg_rest)}")

    rest_count = max(len(fg_rest), len(bg_rest))
    if len(fg_rest) != len(bg_rest):
        print(
            f"SH degree mismatch (fg={len(fg_rest)}, bg={len(bg_rest)} f_rest_* fields); "
            f"promoting both to {rest_count} and zero-padding the lower-degree input."
        )

    merged_dtype = build_merged_dtype(
        fg_vertex.dtype.names or (),
        bg_vertex.dtype.names or (),
        rest_count,
    )

    fg_proj = project_to_dtype(fg_vertex, merged_dtype)
    bg_proj = project_to_dtype(bg_vertex, merged_dtype)
    merged = np.concatenate([fg_proj, bg_proj])
    print(f"Merged vertex count: {len(merged)} ({len(fg_proj)} fg + {len(bg_proj)} bg)")

    element = PlyElement.describe(merged, "vertex")
    PlyData([element], text=False).write(output_path)
    print(f"Saved merged PLY to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a foreground 3DGS PLY with a background 3DGS PLY.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_background_gs.py \\
      --foreground scene01.gs.ply \\
      --background background.gs.ply \\
      --output ./merged_dir
        """,
    )
    parser.add_argument("--foreground", required=True, help="Foreground 3DGS PLY file path")
    parser.add_argument("--background", required=True, help="Background 3DGS PLY file path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (writes merged.gs.ply inside) or output .ply file path",
    )
    parser.add_argument(
        "--output-name",
        default="merged.gs.ply",
        help="Output filename when --output is a directory (default: merged.gs.ply)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.foreground):
        print(f"Error: foreground file not found: {args.foreground}")
        sys.exit(1)
    if not os.path.isfile(args.background):
        print(f"Error: background file not found: {args.background}")
        sys.exit(1)

    output_path = resolve_output_path(args.output, default_name=args.output_name)
    merge_ply(args.foreground, args.background, output_path)


if __name__ == "__main__":
    main()
