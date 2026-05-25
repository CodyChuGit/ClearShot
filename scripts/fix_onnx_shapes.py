#!/usr/bin/env python3
"""
Freeze dynamic input shapes in ONNX models to static [1, 3, 640, 640].

This allows CoreML to compile optimized Metal GPU / Neural Engine programs
instead of falling back to CPU-only execution.
"""

import os
import sys
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto


MODEL_DIR = Path.home() / ".clearshot" / "models"


def fix_model(model_path: Path) -> Path:
    """Fix dynamic input shapes to static 640x640 and save as _static variant."""
    out_path = model_path.with_stem(model_path.stem + "_static")

    print(f"Loading {model_path.name}...")
    model = onnx.load(str(model_path))

    changed = False
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape is None:
            continue

        dims = shape.dim
        # Expecting [batch, channels, height, width]
        if len(dims) == 4:
            names_before = [
                d.dim_param if d.dim_param else str(d.dim_value) for d in dims
            ]

            # Fix batch to 1
            if dims[0].dim_param or dims[0].dim_value != 1:
                dims[0].ClearField("dim_param")
                dims[0].dim_value = 1
                changed = True

            # Fix channels to 3
            if dims[1].dim_param or dims[1].dim_value != 3:
                dims[1].ClearField("dim_param")
                dims[1].dim_value = 3
                changed = True

            # Fix height to 640
            if dims[2].dim_param or dims[2].dim_value != 640:
                dims[2].ClearField("dim_param")
                dims[2].dim_value = 640
                changed = True

            # Fix width to 640
            if dims[3].dim_param or dims[3].dim_value != 640:
                dims[3].ClearField("dim_param")
                dims[3].dim_value = 640
                changed = True

            names_after = [str(d.dim_value) for d in dims]
            if changed:
                print(f"  Input '{inp.name}': [{', '.join(names_before)}] -> [{', '.join(names_after)}]")

    if not changed:
        print(f"  Already static, skipping.")
        return model_path

    onnx.save(model, str(out_path))
    print(f"  Saved: {out_path.name}")

    # Backup original and replace
    backup = model_path.with_suffix(".onnx.bak")
    if not backup.exists():
        shutil.copy2(model_path, backup)
        print(f"  Backup: {backup.name}")
    shutil.copy2(out_path, model_path)
    print(f"  Replaced: {model_path.name}")

    return model_path


def main():
    if not MODEL_DIR.exists():
        print(f"Model directory not found: {MODEL_DIR}")
        sys.exit(1)

    models = list(MODEL_DIR.glob("*.onnx"))
    models = [m for m in models if "_static" not in m.stem and ".bak" not in m.suffixes]

    if not models:
        print("No .onnx models found.")
        sys.exit(1)

    print(f"Found {len(models)} model(s) in {MODEL_DIR}\n")
    for model_path in models:
        fix_model(model_path)
        print()

    print("Done! Restart the server to pick up the fixed models.")


if __name__ == "__main__":
    main()
