# Formats and conversion

| Format | Reader/writer | Extension | Automatic detection |
| --- | --- | --- | --- |
| IBA myQA Accept 6 ASCII | `beamio.rfa.IBAAccept6` | `.asc` | `BeamDataFactory.from_file(path)` |
| RayStation 2025+ measured-curves CSV | `beamio.tps.raystation.Raystation` | `.csv` | Use the class explicitly |

PTW BEAMSCAN/Mephysto (`.mcc`) and SNC SunSCAN 3D (`.snctxt`) are not implemented.
`BeamDataFactory.is_supported_file(path)` checks the extension only; `from_file`
reads file contents to identify a supported reader. Extension discovery
currently returns `['asc']`.

## IBA Accept

The reader recognizes file markers such as `:MSR`, measurement metadata lines
beginning with `%`, and samples written as `= X Y Z Dose`. It maps known
metadata into the shared fields and retains remaining values in `metadata`.

The existing parsing behavior removes sample pairs containing NaN or infinity,
sorts positions, and removes duplicates while keeping the first corresponding
dose. Start coordinates are rounded to 0.2 mm and values within 0.2 mm of zero
are set to zero. The writer reconstructs XYZ coordinates and rounds them to
0.1 mm. Metadata and numerical cleanup mean that files are not byte-for-byte
round trips.

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
metadata key. The automated round-trip tests cover depth, crossline, and inline
scans with synthetic samples. Diagonal geometry and vendor imports need
representative format-specific validation when extending those paths.
