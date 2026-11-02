"""Apply exactly the audited GPL exporter's leaf-reference sentinel correction."""

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
