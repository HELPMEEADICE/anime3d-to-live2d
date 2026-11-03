"""Reproducible staged CLI; all Blender mutations occur in a separate background process."""

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import psutil
from .compile_ir import ROOT


def run(label, cmd, env=None):
    logs = ROOT / "artifacts/logs"
    logs.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    peak = 0
    with (logs / (label + ".log")).open("w", encoding="utf8") as log:
        p = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        process = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                peak = max(
                    peak,
                    sum(
                        c.memory_info().rss
                        for c in [process] + process.children(recursive=True)
                        if c.is_running()
                    ),
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)
    record = {
        "stage": label,
        "seconds": time.perf_counter() - start,
        "peak_process_tree_rss_bytes": peak,
        "exit_code": p.returncode,
        "command": [str(x) for x in cmd],
    }
    perf = ROOT / "artifacts/performance.json"
    old = json.loads(perf.read_text()) if perf.exists() else []
    old.append(record)
    perf.write_text(json.dumps(old, indent=2))
    print(
        f"{label}: {record['seconds']:.2f}s, peak process-tree RSS {peak / 2**20:.1f} MiB",
        flush=True,
    )
    if p.returncode:
        raise RuntimeError(
            f"{label} failed ({p.returncode}); inspect {logs / (label + '.log')}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "all",
            "build",
            "sample",
            "rig",
            "export",
            "validate",
            "visibility",
            "surfaces",
            "report",
            "test",
            "audit",
            "package",
        ],
    )
    p.add_argument("--source", type=Path, default=ROOT / "artifacts/POPY_source.blend")
    p.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
    p.add_argument("--no-cmo", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument(
        "--full-cmo",
        action="store_true",
        help="Attempt the large quality CMO3 (known to exceed a 10 GiB heap); default emits a separately validated balanced editor model",
    )
    args = p.parse_args()
    stages = (
        [
            "build",
            "audit",
            "sample",
            "rig",
            "export",
            "validate",
            "report",
            "visibility",
            "surfaces",
            "test",
            "package",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", r"C:\lib\java")
    env.setdefault("JAVA_OPTS", "-Xmx16g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError")
    if args.stage == "all":
        (ROOT / "artifacts/performance.json").write_text("[]")
    if args.source.exists():
        before = hashlib.sha256(args.source.read_bytes()).hexdigest()
    else:
        before = None
    for stage in stages:
        if stage == "build":
            if args.skip_build:
                continue
            run(
                "rust_build",
                [
                    "cargo",
                    "build",
                    "--manifest-path",
                    "rust/Cargo.toml",
                    "--release",
                    "--locked",
                ],
            )
            run(
                "exporter_build",
                [
                    ROOT / "vendor/psd2live/gradlew.bat",
                    "-p",
                    "exporter",
                    "installDist",
                    "--console=plain",
                ],
                env,
            )
        elif stage == "audit":
            if not args.source.exists():
                raise FileNotFoundError(args.source)
            run(
                "audit",
                [
                    args.blender,
                    "-b",
                    args.source,
                    "--python",
                    "blender/audit.py",
                    "--",
                    ROOT / "artifacts/scene_audit_repeat.json",
                ],
            )
            run(
                "semantics",
                [
                    sys.executable,
                    "-m",
                    "popy.semantics",
                    "artifacts/scene_audit_repeat.json",
                ],
            )
        elif stage == "sample":
            if not args.source.exists():
                raise FileNotFoundError(args.source)
            run(
                "sampling",
                [
                    args.blender,
                    "-b",
                    args.source,
                    "-t",
                    "6",
                    "--python",
                    "blender/sample.py",
                ],
            )
        elif stage == "rig":
            run("geometry_ir", [sys.executable, "-m", "popy.compile_ir"])
            run("occlusion", [sys.executable, "-m", "popy.occlusion"])
        elif stage == "export":
            cmd = [
                ROOT
                / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat",
                "artifacts/rig/POPY.rig.json",
                "output/POPY",
            ]
            if not args.full_cmo:
                cmd += ["--no-cmo"]
            run("export", cmd, env)
            if not args.no_cmo and not args.full_cmo:
                editor_env = env.copy()
                editor_env["POPY_CONFIG"] = str(ROOT / "config/convert_balanced.json")
                run(
                    "editor_geometry",
                    [
                        sys.executable,
                        "-m",
                        "popy.compile_ir",
                        "--out",
                        "artifacts/rig_balanced",
                    ],
                    editor_env,
                )
                run(
                    "editor_occlusion",
                    [
                        sys.executable,
                        "-m",
                        "popy.occlusion",
                        "--rig-dir",
                        "artifacts/rig_balanced",
                    ],
                )
                run(
                    "editor_export",
                    [
                        ROOT
                        / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat",
                        "artifacts/rig_balanced/POPY.rig.json",
                        "output/POPY_balanced",
                    ],
                    env,
                )
                verify_env = env.copy()
                verify_env["JAVA_OPTS"] = os.environ.get(
                    "POPY_CMO_VERIFY_JAVA_OPTS",
                    "-Xmx24g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError",
                )
                run(
                    "editor_roundtrip",
                    [
                        ROOT
                        / "exporter/build/install/popy-cubism-exporter/bin/popy-cubism-exporter.bat",
                        "--verify-cmo",
                        "output/POPY_balanced/POPY.cmo3",
                    ],
                    verify_env,
                )
                run(
                    "editor_ayagami",
                    [
                        ROOT / "rust/target/release/popy-validate.exe",
                        "output/POPY_balanced/POPY.model3.json",
                        "artifacts/cases.json",
                        "artifacts/ayagami_balanced",
                    ],
                )
                run(
                    "editor_comparison",
                    [
                        sys.executable,
                        "-m",
                        "popy.compare",
                        "--rig-dir",
                        "artifacts/rig_balanced",
                        "--runtime-dir",
                        "artifacts/ayagami_balanced",
                        "--out",
                        "artifacts/report_balanced",
                    ],
                )
        elif stage == "validate":
            run(
                "ayagami",
                [
                    ROOT / "rust/target/release/popy-validate.exe",
                    "output/POPY/POPY.model3.json",
                    "artifacts/cases.json",
                    "artifacts/ayagami",
                ],
            )
        elif stage == "visibility":
            run(
                "visibility_regression",
                [
                    sys.executable,
                    "-m",
                    "popy.visibility_regression",
                    "--source",
                    args.source,
                    "--blender",
                    args.blender,
                ],
            )
        elif stage == "report":
            run("comparison", [sys.executable, "-m", "popy.compare"])
        elif stage == "surfaces":
            run(
                "surface_regression",
                [
                    sys.executable,
                    "-m",
                    "popy.surface_runner",
                    "--source",
                    args.source,
                    "--blender",
                    args.blender,
                ],
            )
        elif stage == "package":
            if not args.no_cmo:
                run("delivery", [sys.executable, "-m", "popy.delivery"])
        elif stage == "test":
            run(
                "rust_tests",
                [
                    "cargo",
                    "test",
                    "--manifest-path",
                    "rust/Cargo.toml",
                    "-p",
                    "popy-geometry",
                    "--locked",
                ],
            )
            run(
                "regression",
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            )
    if before:
        after = hashlib.sha256(args.source.read_bytes()).hexdigest()
        assert before == after, "Source snapshot unexpectedly changed"
        (ROOT / "artifacts/source_integrity.json").write_text(
            json.dumps(
                {
                    "source": str(args.source),
                    "sha256_before": before,
                    "sha256_after": after,
                    "unchanged": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
