This executable is GPL-3.0-or-later. It compiles and links the GPL Umamo engine
from the pinned PSD2Live checkout, including its one-line interoperability fix.
See LICENSE in this directory for the full GPL text, and ../docs/THIRD_PARTY.md.
The pinned dependency is fetched locally by setup.ps1; vendor/ is not redistributed
in this source repository.

Original source for the bridge is src/main/kotlin/popy/Export.kt. Its GPL license
does not relicense the vendor MIT format-model module or the neutral IR files.
The geometry converter communicates with this executable through a documented
file IR. Distribution of this exporter and its dependencies must preserve their
own license requirements; process separation alone is not a licensing conclusion
for arbitrary downstream combinations.
