"""
SOAP Note Parser Module

This module provides functionality to parse complete SOAP notes into structured JSON format.
It extracts individual sections (Subjective, Objective, Assessment, Plan) and removes citation tags
while preserving the original raw note.
"""

import re
import json
from typing import Dict, List, Optional
from pathlib import Path


class SOAPNoteParser:
    """
    Parse SOAP notes into structured JSON format with sections separated and citations removed.
    """

    def __init__(self):
        """Initialize the SOAP note parser."""
        # Section headers pattern
        self.section_pattern = re.compile(
            r'^(Subjective|Objective|Assessment|Plan|Speaker Role Mapping):\s*$',
            re.IGNORECASE | re.MULTILINE
        )

        # Citation pattern for removal
        self.citation_pattern = re.compile(r'\s*\[T[^\]]+\]\[SPEAKER [^\]]+\]\s*')

        # Bullet point pattern
        self.bullet_pattern = re.compile(r'^\s*•\s*')

    def remove_citations(self, text: str) -> str:
        """
        Remove citation tags from text.

        Parameters
        ----------
        text : str
            Text with citation tags

        Returns
        -------
        str
            Text with citations removed
        """
        # Remove citation tags
        clean_text = self.citation_pattern.sub('', text)

        # Clean up any extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def parse_section_content(self, content: str, is_bullet_format: bool = True) -> List[str]:
        """
        Parse section content into individual items.

        Parameters
        ----------
        content : str
            Raw section content
        is_bullet_format : bool, default True
            Whether the content uses bullet points

        Returns
        -------
        List[str]
            List of individual content items with citations removed
        """
        if not content.strip():
            return []

        items = []

        if is_bullet_format:
            # Split by bullet points
            lines = content.split('\n')
            current_item = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this is a new bullet point
                if self.bullet_pattern.match(line):
                    # Save previous item if exists
                    if current_item:
                        item_text = ' '.join(current_item)
                        clean_item = self.remove_citations(item_text)
                        if clean_item:
                            items.append(clean_item)

                    # Start new item
                    current_item = [self.bullet_pattern.sub('', line)]
                else:
                    # Continue current item
                    current_item.append(line)

            # Don't forget the last item
            if current_item:
                item_text = ' '.join(current_item)
                clean_item = self.remove_citations(item_text)
                if clean_item:
                    items.append(clean_item)

        else:
            # For paragraph format, split by sentences or paragraphs
            # Remove citations first, then split
            clean_content = self.remove_citations(content)

            # Split by double newlines (paragraphs) or periods for sentences
            if '\n\n' in clean_content:
                items = [item.strip() for item in clean_content.split('\n\n') if item.strip()]
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', clean_content)
                items = [sentence.strip() for sentence in sentences if sentence.strip()]

        return items

    def parse_speaker_mapping(self, content: str) -> Dict[str, str]:
        """
        Parse speaker role mapping section.

        Parameters
        ----------
        content : str
            Speaker mapping content

        Returns
        -------
        Dict[str, str]
            Speaker ID to role mapping
        """
        mapping = {}
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match pattern like "[SPEAKER 0]: doctor"
            match = re.match(r'\[SPEAKER (\d+)\]:\s*(.+)', line)
            if match:
                speaker_id = f"SPEAKER {match.group(1)}"
                role = match.group(2).strip()
                mapping[speaker_id] = role

        return mapping

    def parse_soap_note(self, soap_note: str, detail_level: str = "unknown",
                        organization_style: str = "unknown") -> Dict:
        """
        Parse a complete SOAP note into structured JSON format.

        Parameters
        ----------
        soap_note : str
            Complete SOAP note text
        detail_level : str, default "unknown"
            Detail level used for generation: "brief", "standard", "detailed", "comprehensive"
        organization_style : str, default "unknown"
            Organization style used: "paragraph", "bullet_point"

        Returns
        -------
        Dict
            Structured SOAP note with sections separated and citations removed
        """
        # Initialize result structure
        result = {
            "raw_note": soap_note,
            "parsed_sections": {
                "subjective": [],
                "objective": [],
                "assessment": [],
                "plan": [],
                "speaker_role_mapping": {}
            },
            "metadata": {
                "total_sections": 0,
                "has_citations": bool(self.citation_pattern.search(soap_note)),
                "detail_level": detail_level,
                "organization_style": organization_style if organization_style != "unknown" else (
                    "bullet_point" if "•" in soap_note else "paragraph")
            }
        }

        # Split note into sections
        sections = self.section_pattern.split(soap_note)

        # The first element is usually empty or contains content before first section
        if sections and not sections[0].strip():
            sections = sections[1:]

        # Process sections in pairs (header, content)
        current_section = None

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Check if this is a section header
            if section.lower() in ['subjective', 'objective', 'assessment', 'plan', 'speaker role mapping']:
                current_section = section.lower()
            else:
                # This is section content
                if current_section:
                    if current_section == 'speaker role mapping':
                        result["parsed_sections"]["speaker_role_mapping"] = self.parse_speaker_mapping(section)
                    else:
                        # Determine if it's bullet format
                        is_bullet = "•" in section
                        parsed_content = self.parse_section_content(section, is_bullet)

                        if current_section in result["parsed_sections"]:
                            result["parsed_sections"][current_section] = parsed_content

        # Update metadata
        result["metadata"]["total_sections"] = len([
            v for k, v in result["parsed_sections"].items()
            if k != "speaker_role_mapping" and v
        ])

        return result

    def save_parsed_note(self, parsed_note: Dict, output_file: str):
        """
        Save parsed note to JSON file.

        Parameters
        ----------
        parsed_note : Dict
            Parsed SOAP note structure
        output_file : str
            Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_note, f, indent=2, ensure_ascii=False)

    def batch_parse_notes(self, notes_dict: Dict[str, str], output_dir: str) -> Dict[str, Dict]:
        """
        Parse multiple SOAP notes in batch.

        Parameters
        ----------
        notes_dict : Dict[str, str]
            Dictionary mapping note IDs to SOAP note text
        output_dir : str
            Output directory for JSON files

        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping note IDs to parsed structures
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        for note_id, note_text in notes_dict.items():
            # Parse the note
            parsed_note = self.parse_soap_note(note_text)
            results[note_id] = parsed_note

            # Save individual JSON file
            json_file = output_path / f"{note_id}_parsed.json"
            self.save_parsed_note(parsed_note, str(json_file))

        # Also save combined results
        combined_file = output_path / "all_parsed_notes.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results


# Convenience functions
def parse_soap_note(soap_note: str, detail_level: str = "unknown",
                    organization_style: str = "unknown") -> Dict:
    """
    Convenience function to parse a single SOAP note.

    Parameters
    ----------
    soap_note : str
        Complete SOAP note text
    detail_level : str, default "unknown"
        Detail level used for generation
    organization_style : str, default "unknown"
        Organization style used

    Returns
    -------
    Dict
        Structured SOAP note
    """
    parser = SOAPNoteParser()
    return parser.parse_soap_note(soap_note, detail_level, organization_style)


def parse_and_save_soap_note(soap_note: str, output_file: str,
                             detail_level: str = "unknown",
                             organization_style: str = "unknown") -> Dict:
    """
    Parse and save a SOAP note to JSON file.

    Parameters
    ----------
    soap_note : str
        Complete SOAP note text
    output_file : str
        Output JSON file path
    detail_level : str, default "unknown"
        Detail level used for generation
    organization_style : str, default "unknown"
        Organization style used

    Returns
    -------
    Dict
        Structured SOAP note
    """
    parser = SOAPNoteParser()
    parsed_note = parser.parse_soap_note(soap_note, detail_level, organization_style)
    #parser.save_parsed_note(parsed_note, output_file)
    return parsed_note


if __name__ == "__main__":
    # Example usage and testing
    sample_soap_note = """Subjective:
• Patient reports difficulty urinating, describing the sensation as a "urine pump" [T23,T25][SPEAKER 1]  
• Symptoms worsened after discontinuing terazosin about a month ago [T56-T60][SPEAKER 1]  
• Reports increased urinary frequency [T70-T71][SPEAKER 1]  

Objective:
• Hemorrhoids present on exam but not inflamed [T298-T300][SPEAKER 0]  
• Patient advised to use over-the-counter cream [T302-T303][SPEAKER 0]  

Assessment:
• Primary diagnosis is benign prostatic hyperplasia (BPH) [T69-T73][SPEAKER 0]  
• Hemorrhoids present but not inflamed [T298-T300][SPEAKER 0]  

Plan:
• Restart terazosin using starter blister pack [T361,T375-T378][SPEAKER 0]  
• Follow up with GU for further evaluation [T94,T226][SPEAKER 0]  
• Schedule follow-up in six months [T391-T396][SPEAKER 0]  

Speaker Role Mapping:  
[SPEAKER 0]: doctor  
[SPEAKER 1]: patient"""

    print("Testing SOAP Note Parser...")
    print("=" * 50)

    # Parse the sample note
    parsed_result = parse_soap_note(sample_soap_note, "standard", "bullet_point")

    print("Original note length:", len(sample_soap_note))
    print("Parsed sections:")
    for section, content in parsed_result["parsed_sections"].items():
        if section != "speaker_role_mapping":
            print(f"  {section.title()}: {len(content)} items")
        else:
            print(f"  {section.title()}: {len(content)} speakers")

    print("\nSample parsed content:")
    print("Subjective (first item):", parsed_result)
    print("Speaker mapping:", parsed_result["parsed_sections"]["speaker_role_mapping"])

    # Test saving
    test_output = "test_parsed_note.json"
    parse_and_save_soap_note(sample_soap_note, test_output, "standard", "bullet_point")
    print(f"\nSaved to: {test_output}")

    print("Testing complete!")