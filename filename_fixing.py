""" Utilities for creating and renaming files """

import os
from pathlib import Path


def write_material_file(material_path: Path, texture_filename: str, b_use_color_kd: bool=False):
    if not b_use_color_kd:
        Ka = (1.0, 1.0, 1.00)
        Kd = (1.0, 1.0, 1.0)
        Ks = (0.18, 0.18, 0.18)
    elif "Envy" in os.path.basename(material_path):
        Ka = (0.15, 0.03, 0.03)
        Kd = (0.65, 0.12, 0.10)
        Ks = (0.18, 0.18, 0.18)
    elif "Gala" in os.path.basename(material_path):
        Ka = (0.18, 0.06, 0.03)
        Kd = (0.78, 0.28, 0.12)
        Ks = (0.15, 0.15, 0.15)
    elif "PinkLady" in os.path.basename(material_path):
        Ka = (0.19, 0.07, 0.07)
        Kd = (0.82, 0.30, 0.32)
        Ks = (0.15, 0.18, 0.15)
    elif "CosmicCrisp" in os.path.basename(material_path):
        Ka = (0.10, 0.02, 0.02)
        Kd = (0.42, 0.07, 0.09)
        Ks = (0.15, 0.15, 0.15)
    elif "HoneyCrisp" in os.path.basename(material_path):
        Ka = (0.17, 0.05, 0.03)
        Kd = (0.74, 0.22, 0.14)
        Ks = (0.15, 0.15, 0.15)
    else:
        Ka = (0.15, 0.03, 0.03)
        Kd = (0.65, 0.12, 0.10)
        Ks = (0.18, 0.18, 0.18)

    material_path.write_text(
        "# Blender 5.0.1 MTL File: 'process_apples.blend'\n"
        "# www.blender.org\n"
        "\n"
        "newmtl material_0\n"
        f"Ka {Ka[0]} {Ka[1]} {Ka[2]}\n"
        f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n"
        f"Ks {Ks[0]} {Ks[1]} {Ks[2]}\n"
        "Ns 30.000000\n"
        "Ni 1.4500000\n"
        "d 1.000000\n"
        "illum 2\n"
        f"map_Kd {texture_filename}\n",
        encoding="utf-8",
    )


def _split_filename_into_apple_type_and_number(filename: str) -> tuple[str, int]:
    """Get the apple name part of the filename and the number part.
    Should work correctly even if the filename contains underscores."""
    apple_name = ""
    indx = 0
    for c in filename:
        if c.isdigit() or c == "_":
            break
        apple_name += c
        indx += 1
    
    # Get the number out of the filename
    remainder_filename = filename[indx:]
    number_str = ""
    for c in remainder_filename:
        if c == "_":
            continue
        if not c.isdigit():
            break
        number_str += c

    if number_str == "":
        raise ValueError(f"Filename does not contain a number: {filename}")
    number = int(number_str)
    return apple_name, number


def _apple_name_base(input_name: str) -> str:
    apple_name, number = _split_filename_into_apple_type_and_number(input_name)
    return f"{apple_name}_{number:03d}"


