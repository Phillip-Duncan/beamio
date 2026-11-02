# Formats and conversion

| Format | Reader/writer | Extension | Automatic detection |
| --- | --- | --- | --- |
| IBA myQA Accept 6 ASCII | `beamio.rfa.IBAAccept6` | `.asc` | `BeamDataFactory.from_file(path)` |
| SNC SunSCAN/SunDOSE text | `beamio.rfa.SNCSunScan` | `.snctxt` | `BeamDataFactory.from_file(path)` |
| SNC SunDOSE XML (read only) | `beamio.rfa.SNCSunScan` | `.sncxml` | `BeamDataFactory.from_file(path)` |
| RayStation 2025+ measured-curves CSV | `beamio.tps.raystation.Raystation` | `.csv` | Use the class explicitly |

PTW BEAMSCAN/Mephysto (`.mcc`) is not implemented.
`BeamDataFactory.is_supported_file(path)` checks the extension only; `from_file`
reads file contents to identify a supported reader. Extension discovery
currently returns `['asc', 'snctxt', 'sncxml']`.

## IBA Accept

The reader recognizes file markers such as `:MSR`, measurement metadata lines
beginning with `%`, and samples written as `= X Y Z Dose`. It maps known
metadata into the shared fields and retains remaining values in `metadata`.

The existing parsing behavior removes sample pairs containing NaN or infinity,
sorts positions, and removes duplicates while keeping the first corresponding
dose. Axial start coordinates are rounded to 0.2 mm and values within 0.2 mm of
zero are set to zero. Diagonal scans retain measured XYZ samples and use the
same signed profile projection as SNC, including off-axis profiles. Their
coordinates and doses remain aligned through filtering and deduplication.
The writer reconstructs XYZ coordinates and rounds them to 0.1 mm. Metadata
and numerical cleanup mean that files are not byte-for-byte round trips.

## SNC SunSCAN / SunDOSE

```python
from beamio.rfa import SNCSunScan

scans = SNCSunScan("measurements.snctxt").parse()
scans = SNCSunScan("measurements.sncxml").parse()
raw = SNCSunScan("measurements.sncxml", layer="raw").parse()
earlier = SNCSunScan("measurements.sncxml", layer=2).parse()
scans.save("processed.snctxt")
```

The text reader uses `BEGIN DOSE TABLE` samples and ignores the separate raw
electrometer telemetry table. Summary and scan-header fields populate the
shared model, with remaining fields retained in metadata. Column headers
define coordinate units; native centimetres, field sizes, and SSD are converted
to millimetres. Dose percentages keep their original scale. Parsing preserves
sample order, repeated positions, non-finite dose values, and XYZ samples;
non-finite coordinates are rejected.

Text exports must include `FILE HEADER`, a nonempty `File Version`, and a
nonnegative `File Scan Count`. Both text and XML exports validate their declared
scan count; an empty dataset must explicitly declare zero scans.

Photon rate types map to `FluenceMode` (`Standard`, `FFF`, or `SRS`) for
conversion to RayStation and back to SNC's `Rate Type` when writing text.
Unsupported nonempty photon rate types raise an error rather than defaulting
to a standard beam.

Automatic XML detection respects the encoding declaration/BOM and namespaces,
including UTF-8 and UTF-16 exports. The XML reader processes one scan at a time
and selects the highest
`LayerNumber` by default. `layer="raw"` selects the `RawData` processing layer;
a positive integer selects that layer number. An unavailable layer raises an
error. Readings follow their `Sequence` values. Native XML coordinates and SSD
are in centimetres; field sizes use the exported `MeasurementUnit`.
`measurement.metadata["SNC XML"]` contains scan properties, field/energy/wedge
properties, layer descriptions, and the selected layer's metadata. This API
does not load the underlying raw electrometer channels.

Measured XYZ samples are available as `measurement.coordinates` and through
`reconstruct_xyz()`. This preserves the small off-axis movements in raw layers.
Diagonal `positions` are signed projections onto the profile axis, oriented
toward positive X (positive Y for a vertical profile), independently of scan
direction. Update or clear `coordinates` when editing scan geometry.

The writer exports a processed `.snctxt` dose table for each measurement,
updating common metadata from the shared fields. It does not recreate raw
telemetry or XML processing history. XML writing is not supported. Text output
is tested by reading it back with beamio; re-import into SunDOSE has not yet
been verified.

## RayStation

The CSV format uses semicolons, leading file-header comments, metadata lines,
position/dose pairs, and `End` separators between curves. The reader accepts
decimal commas in numerical fields. The writer maps the shared fields and
uses schema defaults where applicable.

The writer generates a default file header and current export timestamp rather
than reproducing the source header. Energy and SSD have defaults when missing;
set them explicitly when constructing measurements. Supply a field size,
radiation type, measurement type, and startpoint for a meaningful export.
IBA output also needs an endpoint to reconstruct full coordinates. Position
and dose arrays should have equal lengths.

## Example

```python
from beamio.rfa import IBAAccept6
from beamio.tps.raystation import Raystation

source = IBAAccept6("measurements.asc").parse()
converted = source.convert(Raystation)
converted.save("measured_curves.csv")
```

The shared spatial fields use millimetres. Doses retain their input scale.
Conversion copies the shared data; it does not translate every vendor-specific
metadata key. Automated tests cover depth, crossline, inline, and diagonal
profiles, including reversed scans, using synthetic samples. Writers may apply
their own coordinate precision or schema defaults; formats limited to a single
profile axis cannot retain all measured off-axis motion.
