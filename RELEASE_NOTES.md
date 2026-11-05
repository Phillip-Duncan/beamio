# RELEASE NOTES

<!-- version list -->

## v0.8.0 (2026-11-05)

### Features

- **arccheck**: Support measured grids and native movies
  ([`991c2c8`](https://github.com/Phillip-Duncan/beamio/commit/991c2c8b1a85ba15975d53dd8e0ad6027a2bd14a))
  - Add TXT, ACM, and ACML readers and writers with automatic detection - Expose detector frames,
  timestamps, and helical coordinates through Measurements - Preserve acquisition metadata and
  resolve ACML geometry companions - Interpolate missing static sites and preserve zero edits on
  export - Share SNC Patient grid parsing and serialization with SRS MapCHECK - Add regression
  tests, format documentation, and packaging exclusions


## v0.7.0 (2026-11-05)

### Features

- **matrixx**: Support OPG, OPD, and OPW acquisitions
  ([`d4a9444`](https://github.com/Phillip-Duncan/beamio/commit/d4a9444d306bdb12af67c58e349d0cde10e47ce1))
  - Add MatriXX readers and writers with automatic format detection - Expose dose planes and
  snapshot time series through Measurements - Preserve native records, flags, non-dose images, and
  energy units - Resolve workspace paths and reject non-OPD dependencies - Export edited doses at
  native precision and regenerate checksums - Add regression tests, format documentation, and
  packaging exclusions


## v0.6.0 (2026-11-04)

### Features

- **io**: Add native writers and conversion validation
  ([`78d6421`](https://github.com/Phillip-Duncan/beamio/commit/78d64215b422dc4e5e1616cc20808f1936b2e9b4))
  - Write PTW MCC, IC Profiler PRM, and SRS MapCHECK files - Preserve SRS source tables, movie
  channels, and correction records in metadata - Export edited Measurement signals and omit
  interpolated movie pixels - Regenerate SRS checksums and preserve native text encodings - Declare
  format capabilities and warn when dropping unsupported measurements - Add round-trip tests and
  document export and conversion behavior


## v0.5.0 (2026-11-04)

### Features

- **measurement**: Support spatial and temporal acquisitions
  ([`2da6659`](https://github.com/Phillip-Duncan/beamio/commit/2da665919176f84fa73d254e410ab9ec1892d24b))
  - Add SRS MapCHECK dose-map and raw-movie readers with automatic detection - Add dose planes, dose
  volumes, acquisition types, and temporal arrays - Expose IC Profiler frames through shared NumPy
  measurement arrays - Interpolate movie gaps using directional contrasts and perimeter brackets -
  Reject unsupported curve exports before changing output files - Add regression tests,
  documentation, and packaging exclusions


## v0.4.0 (2026-11-04)

### Features

- **icprofiler**: Add PRM acquisitions and profile analysis
  ([`81777a3`](https://github.com/Phillip-Duncan/beamio/commit/81777a31b5ce5f1bc89216534bd7fa26bdd274fe))
  - Read PRM acquisitions with embedded calibration and four-axis geometry - Expose raw frames,
  calibrated detector time series, and beam-centre data - Add standalone profile helpers with
  InflectionPoint and optional FWHM - Attempt unvalidated PRM versions with a warning - Validate
  acquisition totals and reject ambiguous field edges - Add regression tests, documentation, and
  packaging exclusions


## v0.3.0 (2026-11-03)

### Features

- **ptw**: Add Mephysto MCC water-tank reader
  ([`5f2bf96`](https://github.com/Phillip-Duncan/beamio/commit/5f2bf96d5722b9b3e0373caf10dbfc2f76017ee2))
  - Read photon and electron axial scans with reference readings - Preserve beam metadata and
  displaced field edges - Detect MCC files reliably despite conflicting comment text - Add
  regression tests, documentation, and packaging exclusions


## v0.2.0 (2026-11-02)

### Features

- **snc**: Add SunSCAN and SunDOSE beam data support
  ([`8919e00`](https://github.com/Phillip-Duncan/beamio/commit/8919e00f90a1dd974ee01e5c79954e448b169602))
  - Read SNC text/XML exports with processing-layer selection - Write processed text and detect XML
  encodings and namespaces - Preserve measured XYZ coordinates and photon fluence modes - Correct
  off-axis diagonal IBA round trips - Reject incomplete headers and invalid scan counts - Add
  regression tests, format documentation, and packaging safeguards


## v0.1.1 (2026-11-02)

### Bug Fixes

- **docs**: Simplify release documentation
  ([`9c71b24`](https://github.com/Phillip-Duncan/beamio/commit/9c71b24f6363b352b5356d68ba943cf29b916a37))


## v0.1.0 (2026-11-02)

### Bug Fixes

- Fixed erroneous imports
  ([`a44dcf1`](https://github.com/Phillip-Duncan/beamio/commit/a44dcf1a42cddd106c21e1b0e94960a17857789f))

- Remove duplicate positions (and their corresponding dose values) sometimes present in IBAAccept6
  data
  ([`a0c0bb3`](https://github.com/Phillip-Duncan/beamio/commit/a0c0bb36d0ed2804d36fdb1cd4c49242a7ddcbe0))

- Round iba start points within 0.2 mm to 0.0. This appears to be the realistic combined accuracy of
  the BP2.
  ([`4662b35`](https://github.com/Phillip-Duncan/beamio/commit/4662b35cf832a32fda13b1ba35113d7786a77734))
