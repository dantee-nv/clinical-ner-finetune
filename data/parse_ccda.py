"""Minimal C-CDA XML parser for proof-of-concept clinical NLP workflows.

This parser intentionally supports only a tiny subset of C-CDA-like structure:
ClinicalDocument/component/structuredBody/component/section/{title,text}.
It is designed for demonstration, not full C-CDA compliance.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    import xmltodict
except ImportError:  # pragma: no cover - environment-specific
    xmltodict = None


def clean_text(text: str) -> str:
    """Normalize whitespace for human-readable section text."""
    return re.sub(r"\s+", " ", text).strip()


def _as_list(value: Any) -> list[Any]:
    """Return value as a list to simplify single-item vs list handling."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _strip_namespaces(value: Any) -> Any:
    """Recursively remove XML namespace prefixes from dictionary keys."""
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = key.split(":")[-1]
            normalized[normalized_key] = _strip_namespaces(child)
        return normalized
    if isinstance(value, list):
        return [_strip_namespaces(item) for item in value]
    return value


def _flatten_text(value: Any) -> str:
    """Flatten nested XML-to-dict text nodes into a plain text string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_flatten_text(item) for item in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, child in value.items():
            if key.startswith("@"):
                continue
            parts.append(_flatten_text(child))
        return " ".join(part for part in parts if part)
    return str(value)


def _strip_tag_namespace(tag: str) -> str:
    """Remove namespace URI wrappers from ElementTree tag names."""
    if "}" in tag:
        return tag.split("}", maxsplit=1)[1]
    return tag


def _parse_with_elementtree(xml_content: str) -> list[dict[str, str]]:
    """Fallback parser when xmltodict is unavailable."""
    root = ET.fromstring(xml_content)
    sections: list[dict[str, str]] = []

    for element in root.iter():
        if _strip_tag_namespace(element.tag) != "section":
            continue

        title = ""
        text = ""
        for child in element.iter():
            tag = _strip_tag_namespace(child.tag)
            if tag == "title":
                title = clean_text(" ".join(child.itertext()))
            elif tag == "text":
                text = clean_text(" ".join(child.itertext()))

        if not title and not text:
            continue

        sections.append(
            {
                "title": title or "Untitled Section",
                "text": text,
            }
        )

    return sections


def parse_ccda_sections(xml_content: str) -> list[dict[str, str]]:
    """Parse section title/text pairs from a minimal C-CDA XML document."""
    if xmltodict is None:
        return _parse_with_elementtree(xml_content)

    parsed = _strip_namespaces(xmltodict.parse(xml_content))

    clinical_document = parsed.get("ClinicalDocument", {})
    component = clinical_document.get("component", {})
    structured_body = component.get("structuredBody", {})

    sections: list[dict[str, str]] = []
    for body_component in _as_list(structured_body.get("component")):
        section = body_component.get("section", {}) if isinstance(body_component, dict) else {}
        title = clean_text(_flatten_text(section.get("title")))
        text = clean_text(_flatten_text(section.get("text")))

        if not title and not text:
            continue

        sections.append(
            {
                "title": title or "Untitled Section",
                "text": text,
            }
        )

    return sections


def sections_to_plain_text(sections: list[dict[str, str]]) -> str:
    """Render parsed sections into a readable plain-text report."""
    rendered: list[str] = []
    for section in sections:
        rendered.append(f"[{section['title']}]")
        rendered.append(section["text"])
        rendered.append("")
    return "\n".join(rendered).strip() + "\n"


def parse_file(input_xml: Path) -> list[dict[str, str]]:
    """Read an XML file and return parsed C-CDA sections."""
    xml_content = input_xml.read_text(encoding="utf-8")
    return parse_ccda_sections(xml_content)


def save_outputs(sections: list[dict[str, str]], output_txt: Path, output_json: Path) -> None:
    """Persist parsed outputs for inspection and debugging."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_txt.write_text(sections_to_plain_text(sections), encoding="utf-8")
    output_json.write_text(json.dumps(sections, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for C-CDA parsing script."""
    parser = argparse.ArgumentParser(description="Parse a synthetic C-CDA XML sample into plain text.")
    parser.add_argument(
        "--input_xml",
        type=Path,
        default=Path("data/raw/sample_ccda.xml"),
        help="Path to synthetic C-CDA XML file.",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=Path("data/interim/parsed_ccda.txt"),
        help="Where to write parsed plain-text sections.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("data/interim/parsed_ccda_sections.json"),
        help="Where to write parsed sections as JSON.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    sections = parse_file(args.input_xml)
    save_outputs(sections, args.output_txt, args.output_json)

    print(f"Parsed {len(sections)} sections from {args.input_xml}")
    print(f"Plain text output: {args.output_txt}")
    print(f"JSON output: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
