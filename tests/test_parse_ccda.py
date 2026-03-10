"""Unit tests for synthetic C-CDA parsing."""

from __future__ import annotations

from data.parse_ccda import parse_ccda_sections, sections_to_plain_text


def test_parse_ccda_sections_extracts_titles_and_text() -> None:
    xml = """
    <hl7:ClinicalDocument xmlns:hl7="urn:hl7-org:v3">
      <hl7:component>
        <hl7:structuredBody>
          <hl7:component>
            <hl7:section>
              <hl7:title>Assessment</hl7:title>
              <hl7:text>Patient has asthma exacerbation.</hl7:text>
            </hl7:section>
          </hl7:component>
        </hl7:structuredBody>
      </hl7:component>
    </hl7:ClinicalDocument>
    """
    sections = parse_ccda_sections(xml)

    assert len(sections) == 1
    assert sections[0]["title"] == "Assessment"
    assert "asthma exacerbation" in sections[0]["text"].lower()


def test_parse_ccda_sections_skips_completely_empty_section() -> None:
    xml = """
    <ClinicalDocument>
      <component>
        <structuredBody>
          <component>
            <section>
              <title></title>
              <text></text>
            </section>
          </component>
        </structuredBody>
      </component>
    </ClinicalDocument>
    """
    sections = parse_ccda_sections(xml)
    assert sections == []


def test_sections_to_plain_text_contains_section_headers() -> None:
    sections = [
        {"title": "Problems", "text": "Diabetes mellitus."},
        {"title": "Plan", "text": "Order A1c."},
    ]
    rendered = sections_to_plain_text(sections)

    assert "[Problems]" in rendered
    assert "Diabetes mellitus." in rendered
    assert "[Plan]" in rendered
