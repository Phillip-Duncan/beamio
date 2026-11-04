# Architecture

beamio is a headless Python library with a `src` layout and one runtime
dependency, NumPy. Importing the package does not read files or start services.

| Area | Responsibility |
| --- | --- |
| `src/beamio/common/measurement.py` | NumPy samples, measurement and radiation enums, shared metadata, and coordinate reconstruction |
| `src/beamio/profile.py` | Public vendor-independent profile edges, centres, and widths from one X/Y profile |
| `src/beamio/common/beamdata.py` | Abstract reader/writer interface and conversion through independent copies |
| `src/beamio/rfa/iba.py` | IBA myQA Accept 6 ASCII parsing and writing |
| `src/beamio/rfa/snc.py` | SNC text parsing/writing and shared-model mapping; `_snc_xml.py` reads XML layers |
| `src/beamio/rfa/factory.py` | Content-based detection and supported-extension discovery |
| `src/beamio/rfa/ptw.py` | PTW Mephysto MCC water-tank parsing and shared-model mapping |
| `src/beamio/rfa/icprofiler.py` | IC PROFILER PRM frames, calibration, and four-axis profiles |
| `src/beamio/rfa/_icprofiler_series.py` | IC Profiler time-series results and detector signals, delegating profile analysis to `beamio.profile` |
| `src/beamio/tps/raystation.py` | RayStation measured-curves CSV parsing and writing |

## Shared model

`Measurement` holds `positions` and `doses` as one-dimensional NumPy arrays.
`MeasurementType` identifies depth, crossline, inline, and diagonal scans.
`RadiationType` identifies photons, electrons, and protons. Optional fields
include energy, SSD, field size, wedge angle, and the scan start/end coordinates.
Format-specific values remain in `measurement.metadata`.

The public `beamio.profile` module accepts position/dose arrays and depends
only on NumPy. It never imports readers or the shared model. Users pass
measurement arrays explicitly when requesting analysis. IC Profiler
time-series methods select valid frames, apply their relative signal filter,
and call the same functions with each frame's X/Y samples. Analysis does not
mutate acquired arrays or metadata.

An optional `coordinates` array stores measured XYZ samples in millimetres,
including off-axis acquisition motion. `reconstruct_xyz()` returns these
samples when present and checks that they still match positions and endpoints.
After editing geometry, update the coordinates too, or set `coordinates = None`
to reconstruct an ideal profile from positions and endpoints.

`BeamData` holds a list of measurements and file-level metadata. Subclasses
implement `parse()` and `save(path)`. Parsing populates the instance and returns
it, so `IBAAccept6(path).parse()` is the normal reader pattern.

`beam.convert(TargetClass)` constructs a new target instance and deep-copies
the measurements and file metadata without recalibrating or normalizing doses.
Format readers and writers translate their units
to and from the shared millimetre model. `save()` applies the target
writer's schema and defaults; format-specific metadata may not transfer.

## Adding a format

Put shared model behavior in `common`, vendor measurement readers in `rfa`, and
treatment-planning interfaces in `tps`. Shared code must not import vendor
implementations. Implement a `BeamData` subclass and add corresponding tests
under the matching source area. Add complete conversions under
`tests/integration`. Extend the factory's detection rules only when a reader
has an unambiguous signature and is implemented.

## Package metadata

`pyproject.toml` owns the build configuration, dependencies, version, tool
settings, and semantic release policy. Runtime `__version__` uses installed
metadata, with a `0.0.0.dev` fallback for an uninstalled source
tree. `MANIFEST.in` includes documentation, synthetic tests, release templates,
and the GPLv2 license in the source archive. The wheel contains the package and
its distribution metadata.
