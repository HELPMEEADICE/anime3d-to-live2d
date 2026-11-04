This executable is GPL-3.0-or-later. It compiles and links the GPL Umamo engine
from the pinned PSD2Live checkout, including the documented draw-leaf sentinel and CMO3 Glue lookup corrections.
See LICENSE in this directory for the full GPL text, and ../docs/THIRD_PARTY.md.
The pinned dependency is fetched locally by setup.ps1; vendor/ is not redistributed
in this source repository.

Original source for the bridge is src/main/kotlin/popy/Export.kt. Its GPL license
does not relicense the vendor MIT format-model module or the neutral IR files.
The geometry converter communicates with this executable through a documented
file IR. Distribution of this exporter and its dependencies must preserve their
own license requirements; process separation alone is not a licensing conclusion
for arbitrary downstream combinations.

The GPL patch under patches/ caches CMO3 Glue lookup. It is a local PSD2Live
change, applied by popy/setup_exporter.py; no upstream submission is made.

Export.kt also lowers explicit Warp parents and non-offscreen scalar Parts,
including their organizational ownership and stable render-group occurrences.
These additions remain GPL-3.0-or-later within this executable.
