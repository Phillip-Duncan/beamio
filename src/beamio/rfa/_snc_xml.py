"""Read SunDOSE XML exports, retaining one processing layer per scan."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def values(element: ET.Element) -> dict[str, str]:
    return {
        prop.attrib["name"]: prop.get("value", "")
        for prop in element.findall("p")
        if prop.get("exporttype") == "Value" and "name" in prop.attrib
    }


def container(scan: ET.Element, name: str) -> dict[str, str]:
    parent = scan.find(f"p[@name='{name}']")
    return values(parent[0]) if parent is not None and len(parent) else {}


def number(value: str) -> float:
    return float(value.strip().replace(",", "."))


def scan_data(scan: ET.Element, selection: str | int):
    properties = values(scan)
    layers = [(values(layer), layer) for layer in scan.findall(".//ScanLayer")]
    if not layers:
        raise ValueError("SNC XML scan has no processing layers.")
    if selection == "raw":
        candidates = [item for item in layers if item[0].get("Type") == "RawData"]
    elif isinstance(selection, int):
        candidates = [item for item in layers if int(item[0]["LayerNumber"]) == selection]
    else:
        candidates = layers
    if not candidates:
        raise ValueError(f"SNC XML scan has no layer matching {selection!r}.")
    selected, element = max(candidates, key=lambda item: int(item[0]["LayerNumber"]))
    readings = element.findall("./p[@name='Readings']/ScanReadingCollection/ScanReading")
    if not readings:
        raise ValueError("Selected SNC XML layer has no dose readings.")

    points, doses = [], []
    readings.sort(key=lambda reading: int(values(reading)["Sequence"]))
    for reading in readings:
        point = container(reading, "Point")
        points.append([number(point[axis]) * 10 for axis in ("X", "Y", "Z")])
        doses.append(number(values(reading)["RelativeDose"]))

    field = container(scan, "FieldSize")
    energy = container(scan, "Energy")
    wedge = container(scan, "Wedge")
    collimation = container(scan, "Collimation")
    units = field.get("MeasurementUnit", "Centimeter")
    if units not in {"Centimeter", "Millimeter"}:
        raise ValueError(f"Unsupported SNC XML field-size unit: {units!r}")
    to_cm = 0.1 if units == "Millimeter" else 1.0
    metadata = {
        "Scan Id": properties.get("ScanId", ""),
        "Scan Date": properties.get("MeasurementDate", ""),
        "Comments": properties.get("Comments", ""),
        "Scan Type": properties.get("Direction") or properties.get("Type", ""),
        "Beam Type": energy.get("Source") or field.get("EnergySource", ""),
        "Energy (MV / MeV)": energy.get("Amount", ""),
        "Rate Type": energy.get("DoseRate", ""),
        "Source to Surface Distance (cm)": properties.get("SourceSurfaceDistance", ""),
        "Field Size X (cm)": str(number(field["X"]) * to_cm) if field.get("X") else "",
        "Field Size Y (cm)": str(number(field["Y"]) * to_cm) if field.get("Y") else "",
        "Wedge Type": wedge.get("Type", ""),
        "Wedge Angle (degrees)": wedge.get("Angle", ""),
        "Collimator Angle (degrees)": collimation.get("Angle", ""),
        "Action": (
            selected.get("Type") if selected.get("Type") in {"Normalize", "Scale"} else "Standard"
        ),
        "Is PDI to PDD": str(
            any(
                info.get("Type") == "PercentIonizationToPdd"
                and int(info["LayerNumber"]) <= int(selected["LayerNumber"])
                for info, _ in layers
            )
        ),
        "SNC XML": {
            "Scan": properties,
            "FieldSize": field,
            "Energy": energy,
            "Wedge": wedge,
            "Collimation": collimation,
            "Layers": [info for info, _ in layers],
            "SelectedLayer": selected,
        },
    }
    return metadata, np.asarray(points, dtype=float), np.asarray(doses, dtype=float)


def read_xml(path: Path, selection: str | int):
    """Stream by scan; raw electrometer telemetry is not part of the dose model."""
    metadata, scans, stack = {}, [], []
    try:
        with path.open("rb") as source:
            for event, element in ET.iterparse(source, events=("start", "end")):
                if event == "start":
                    element.tag = element.tag.rsplit("}", 1)[-1]
                    if not stack:
                        if element.tag != "snc_fileexport":
                            raise ValueError("Not an SNC XML export: expected snc_fileexport root.")
                        metadata.update(
                            {
                                "File Export Version": element.get("version", ""),
                                "File Date": element.get("date", ""),
                            }
                        )
                    elif element.tag == "scans" and len(stack) == 1:
                        metadata["File Scan Count"] = element.get("count", "")
                    stack.append(element)
                    continue
                parent = stack[-2] if len(stack) > 1 else None
                if element.tag == "Scan":
                    scans.append(scan_data(element, selection))
                    element.clear()
                    if parent is not None:
                        parent.remove(element)
                elif element.tag == "ScanRawDataItem":
                    element.clear()
                    if parent is not None:
                        parent.remove(element)
                elif len(stack) == 2 and element.tag in {"notes", "entities"}:
                    if element.tag == "notes":
                        metadata["Notes"] = element.text or ""
                    else:
                        metadata["SNC XML Entities"] = {
                            child.tag: values(child)
                            for child in element.iter()
                            if child.tag != "p" and child.findall("p")
                        }
                    element.clear()
                stack.pop()
    except (ET.ParseError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid SNC XML file {path.name}: {error}") from error
    if "File Scan Count" not in metadata:
        raise ValueError(f"SNC XML file {path.name} is missing its scans collection.")
    return metadata, scans
