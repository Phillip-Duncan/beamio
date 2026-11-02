# RELEASE NOTES

<!-- version list -->

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
