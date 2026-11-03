"""Apply the audited local GPL exporter corrections to the pinned checkout."""

import subprocess

from .compile_ir import ROOT

path = (
    ROOT
    / "vendor/psd2live/umamo/src/main/kotlin/org/umamo/interop/moc3/export/Moc3RenderOrderLowering.kt"
)
old = "RenderOrderChild(kind = 0, index = drawableIndex, groupIndex = 0)"
new = "RenderOrderChild(kind = 0, index = drawableIndex, groupIndex = -1)"
text = path.read_text(encoding="utf8")
if old in text:
    path.write_text(text.replace(old, new), encoding="utf8")
elif new not in text:
    raise RuntimeError("Unknown exporter source revision: refusing a broad patch")
print("Audited draw-leaf sentinel correction is present")
repo = ROOT / "vendor/psd2live"
for name in ["cmo_glue_lookup.patch", "moc_position_dedup.patch"]:
    patch = ROOT / "exporter/patches" / name
    reverse = subprocess.run(
        ["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)],
        capture_output=True,
        check=False,
    )
    if reverse.returncode:
        forward = subprocess.run(
            ["git", "-C", str(repo), "apply", "--check", str(patch)],
            capture_output=True,
            check=False,
        )
        if forward.returncode:
            raise RuntimeError(
                "Unknown Cubism format source revision; refusing patch " + name
            )
        subprocess.run(["git", "-C", str(repo), "apply", str(patch)], check=True)
    print("Audited correction is present: " + name)
