"""Synthetic examples of SNC's text and property-based XML schemas."""

import xml.etree.ElementTree as ET

SNC_TEXT = """Tab-Delimited Scan Output
FILE HEADER
File Export Version\t4.1.0.208\t
File Version\t6.0\t
File Scan Count\t1\t
BEGIN SCAN
Summary Comments\t\t
Summary Beam Type\tPhoton\t
Beam Type\tPhoton\t
Summary Energy (MV/MeV)\t6\t
Energy (MV / MeV)\t6\t
Rate Type\tFF\t
Summary FieldSize X (cm)\t10\t
Summary FieldSize Y (cm)\t20\t
Summary Wedge Angle (degrees)\t0\t
Summary Scan Type\tCrossline\t
Is PDI to PDD\tFalse\t
BEGIN DOSE TABLE
Action\tNormalize\t
\tX (cm)\tY (cm)\tZ (cm)\tRelative Dose (%)
\t0.5\t0\t1\t55
\t0\t0\t1\t100
\t-0.5\t0\t1\t65
END DOSE TABLE
SCAN HEADER
Source to Surface Distance (cm)\t100\t
Collimator Angle (degrees)\t12\t
Gantry Angle (degrees)\t90\t
Scan Id\t1\t
Custom Metadata\tpreserved\t
BEGIN RAW DATA
\tSample\tElectrometer Field\tElectrometer Reference\tUnprocessed Value
\t999\t999\t999\t999
END RAW DATA
END SCAN
"""


def xml_export(
    *, count=1, field_unit="Centimeter", direction="Crossline", namespace=False, rate_type="FF"
):
    root = ET.Element("snc_fileexport", version="4.00", date="01/01/2026 12:00:00")
    if namespace:
        root.set("xmlns", "urn:synthetic-snc")
    ET.SubElement(root, "notes").text = "Synthetic test export"
    scans = ET.SubElement(root, "scans", count=str(count))

    def prop(parent, name, value):
        ET.SubElement(parent, "p", name=name, exporttype="Value", value=str(value))

    def container(parent, name, tag):
        return ET.SubElement(ET.SubElement(parent, "p", name=name, exporttype="Container"), tag)

    for index in range(count):
        entities = ET.SubElement(ET.SubElement(scans, "scan"), "entities", RadiationDevice="1")
        scan = ET.SubElement(entities, "Scan", exporttype="Container")
        for key, value in {
            "ScanId": index + 1,
            "Direction": direction,
            "Type": "Profile",
            "SourceSurfaceDistance": "100.00",
            "Comments": "synthetic",
        }.items():
            prop(scan, key, value)
        field = container(scan, "FieldSize", "Field")
        for key, value in {"X": 10, "Y": 20, "MeasurementUnit": field_unit}.items():
            prop(field, key, value)
        energy = container(scan, "Energy", "Energy")
        prop(energy, "Source", "Photon")
        prop(energy, "Amount", 6)
        prop(energy, "DoseRate", rate_type)
        wedge = container(scan, "Wedge", "ScanWedge")
        prop(wedge, "Angle", 0)
        prop(wedge, "Type", "Open")
        layers = container(scan, "Layers", "ScanLayerCollection")
        # XML collection order need not match the explicit processing-layer number.
        for layer_number, kind, doses in [
            (3, "Normalize", [55, 100, 65]),
            (1, "RawData", [0.1, 0.2, 0.3, 0.4]),
        ]:
            layer = ET.SubElement(layers, "ScanLayer", exporttype="Container")
            prop(layer, "LayerNumber", layer_number)
            prop(layer, "Type", kind)
            prop(layer, "Details", "synthetic layer")
            readings = container(layer, "Readings", "ScanReadingCollection")
            for sequence, dose in reversed(list(enumerate(doses, start=1))):
                reading = ET.SubElement(readings, "ScanReading", exporttype="Container")
                prop(reading, "Sequence", sequence)
                prop(reading, "RelativeDose", dose)
                point = container(reading, "Point", "DataPoint")
                for axis, coordinate in zip(("X", "Y", "Z"), (1 - sequence * 0.5, 0, 1)):
                    prop(point, axis, coordinate)
        telemetry = container(scan, "RawData", "ScanRawDataItemCollection")
        prop(ET.SubElement(telemetry, "ScanRawDataItem"), "UnprocessedValue", 999)
    return ET.tostring(root, encoding="unicode")
