"""Reproducible depth-visible eye, source seam and continuous-motion regression."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .cli import run
from .compile_ir import ROOT, jsonwrite
from .visibility_cases import cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "output/POPY/POPY.model3.json")
    p.add_argument("--rig", type=Path, default=ROOT / "artifacts/rig/POPY.rig.json")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/visibility")
    p.add_argument("--source", type=Path, default=ROOT / "artifacts/POPY_source.blend")
    p.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
    p.add_argument(
        "--reuse-reference",
        action="store_true",
        help="Reuse the existing independently rendered references after checking their inventory",
    )
    args = p.parse_args()
    args.out = args.out.resolve()
    args.model = args.model.resolve()
    args.rig = args.rig.resolve()
    case_file = ROOT / "artifacts/visibility_cases.json"
    jsonwrite(case_file, cases())
    reference = ROOT / "artifacts/visibility_reference"
    if not args.reuse_reference:
        run(
            "visibility_reference",
            [
                args.blender,
                "-b",
                args.source,
                "-t",
                "6",
                "--python",
                "blender/sample.py",
                "--",
                "--references-only",
                "--eye-masks",
                "--case-file",
                case_file,
                "--case-output",
                ROOT / "artifacts/visibility_reference_cases.json",
                "--reference-dir",
                reference,
            ],
        )
    for case in cases():
        for suffix in [".png", "_eyes.png"]:
            if not (reference / (case["name"] + suffix)).exists():
                raise FileNotFoundError(reference / (case["name"] + suffix))
        for obj in json.loads((ROOT / "artifacts/samples/manifest.json").read_text())[
            "objects"
        ]:
            path = reference / (case["name"] + "__" + obj["name"] + ".npy")
            if not path.exists():
                raise FileNotFoundError(path)
    from .visibility_validation import assess, eye_palette

    palette = eye_palette(args.model, args.out / "palette")
    renderer = ROOT / "rust/target/release/popy-validate.exe"
    run("visibility_runtime", [renderer, args.model, case_file, args.out / "runtime"])
    run(
        "visibility_eye_runtime",
        [renderer, palette, case_file, args.out / "eyes", "--images-only"],
    )
    result = assess(
        args.rig, args.out / "runtime", args.out / "eyes", args.out / "assessment"
    )
    model = json.loads(args.model.read_text())
    moc = args.model.parent / model["FileReferences"]["Moc"]
    jsonwrite(
        args.out / "provenance.json",
        {
            "moc_sha256": hashlib.sha256(moc.read_bytes()).hexdigest(),
            "rig_sha256": hashlib.sha256(args.rig.read_bytes()).hexdigest(),
            "cases_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
            "reference_reused": args.reuse_reference,
            "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
            "summary": result["summary"],
        },
    )
    run(
        "visibility_report",
        [sys.executable, "-m", "popy.visibility_report", "--current", args.out],
    )


if __name__ == "__main__":
    main()
