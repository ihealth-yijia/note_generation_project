"""
Transcript Indexer Module

This module provides functionality to convert raw medical transcripts to indexed format
with speaker tags and turn numbers. It supports multiple transcript formats and provides
both class-based and function-based interfaces for easy integration.
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class TranscriptIndexer:
    """
    Convert raw medical transcripts to indexed format with speaker tags and turn numbers.
    Supports multiple transcript formats including DOCTOR/PATIENT and Speaker X: styles.
    """

    def __init__(self):
        """Initialize the transcript indexer."""
        # Common speaker identification patterns
        self.doctor_keywords = ['doctor', 'dr', 'physician', 'md', 'provider', 'clinician']
        self.patient_keywords = ['patient', 'pt', 'client']

    def read_transcript(self, file_path: str) -> str:
        """
        Read transcript text file and return content as string.

        Parameters
        ----------
        file_path : str
            Path to the transcript file

        Returns
        -------
        str
            Raw transcript content

        Raises
        ------
        Exception
            If file cannot be read
        """
        try:
            file_path = Path(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                raise Exception(f"Error reading transcript file {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error reading transcript file {file_path}: {e}")

    def process_doctor_patient_transcript(self, text: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Process transcript text in DOCTOR/PATIENT style and return indexed lines.

        This method handles transcripts where speakers are identified with labels like:
        - DOCTOR
        - PATIENT
        - DOCTOR B552
        - PATIENT A123

        Parameters
        ----------
        text : str
            Raw transcript text

        Returns
        -------
        Tuple[List[str], Dict[str, str]]
            Tuple of (indexed_lines, speaker_mapping)
            - indexed_lines: List of strings in format "[T#] [SPEAKER #]: utterance"
            - speaker_mapping: Dict mapping original labels to SPEAKER IDs

        Example
        -------
        >>> indexer = TranscriptIndexer()
        >>> text = '''DOCTOR
        ... How are you feeling today?
        ... PATIENT
        ... I'm feeling much better.'''
        >>> lines, mapping = indexer.process_doctor_patient_transcript(text)
        >>> print(lines[0])
        '[T1] [SPEAKER 0]: How are you feeling today?'
        """
        speaker_map = {}
        speaker_counter = 0
        current_speaker = None
        t_counter = 1
        output_lines = []

        # Remove metadata lines and clean up
        cleaned_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not re.match(
                r"^(Date|Time|DOCTOR OUT|PATIENT LEAVES|DD:|DT:|INFOPRO|TRANSACTION)",
                line, re.IGNORECASE
            )
        ]

        for line in cleaned_lines:
            # Match speaker lines like "DOCTOR B552" or "PATIENT"
            speaker_match = re.match(r"^(DOCTOR|PATIENT)(?:\s+\S+)?$", line, re.IGNORECASE)

            if speaker_match:
                speaker_label = speaker_match.group(1).upper()
                if speaker_label not in speaker_map:
                    speaker_map[speaker_label] = f"SPEAKER {speaker_counter}"
                    speaker_counter += 1
                current_speaker = speaker_map[speaker_label]

            elif current_speaker:
                # Clean the line and split into utterances
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Split line into sentences for better granularity
                # Handle common sentence endings and line breaks
                utterances = re.split(r'(?<=[.!?])\s+|\n+', clean_line)

                for utt in utterances:
                    utt = utt.strip()
                    if utt:
                        # Remove any remaining special characters that might interfere
                        utt = re.sub(r'[@#$%^&*()]+', '', utt)
                        if utt:  # Make sure we still have content after cleaning
                            output_lines.append(f"[T{t_counter}] [{current_speaker}]: {utt}")
                            t_counter += 1

        return output_lines, speaker_map

    def process_deepgram_transcript(self, text: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Process Deepgram-style transcript with Speaker X: format.

        This method handles transcripts where speakers are identified as:
        - Speaker 0: utterance text
        - Speaker 1: utterance text

        Parameters
        ----------
        text : str
            Raw transcript text

        Returns
        -------
        Tuple[List[str], Dict[str, str]]
            Tuple of (indexed_lines, speaker_mapping)

        Example
        -------
        >>> indexer = TranscriptIndexer()
        >>> text = '''Speaker 0: How are you feeling today?
        ... Speaker 1: I'm feeling much better, thanks.'''
        >>> lines, mapping = indexer.process_deepgram_transcript(text)
        >>> print(lines[0])
        '[T1] [SPEAKER 0]: How are you feeling today?'
        """
        # Split on speaker boundaries while preserving speaker labels
        speaker_blocks = re.split(r'\n(?=Speaker \d+:)', text.strip())
        numbered_lines = []
        line_counter = 1
        all_speakers = set()

        for block in speaker_blocks:
            # Match speaker ID and content
            match = re.match(r'Speaker (\d+):\s*(.*)', block, re.DOTALL)
            if not match:
                continue

            speaker_id = match.group(1)
            content = match.group(2).strip()
            all_speakers.add(speaker_id)

            if not content:
                continue

            # Split content into sentences for better granularity
            sentences = re.split(r'(?<=[.!?])\s+|\n+', content)

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    # Clean up any remaining artifacts
                    sentence = re.sub(r'[@#$%^&*()]+', '', sentence)
                    if sentence:
                        line = f"[T{line_counter}] [SPEAKER {speaker_id}]: {sentence}"
                        numbered_lines.append(line)
                        line_counter += 1

        # Create speaker mapping based on discovered speakers
        speaker_map = {f"SPEAKER {i}": f"SPEAKER {i}" for i in sorted(all_speakers)}

        return numbered_lines, speaker_map

    def process_transcript_file(self, file_path: str, format_type: str = "doctor_patient") -> Tuple[
        str, Dict[str, str]]:
        """
        Process a transcript file and return indexed transcript string.

        Parameters
        ----------
        file_path : str
            Path to transcript file
        format_type : str, default "doctor_patient"
            Transcript format: "doctor_patient" or "deepgram"

        Returns
        -------
        Tuple[str, Dict[str, str]]
            Tuple of (indexed_transcript_string, speaker_mapping)

        Raises
        ------
        ValueError
            If format_type is not recognized
        Exception
            If file processing fails
        """
        # Read the file
        raw_text = self.read_transcript(file_path)

        # Process based on format type
        if format_type == "doctor_patient":
            indexed_lines, speaker_mapping = self.process_doctor_patient_transcript(raw_text)
        elif format_type == "deepgram":
            indexed_lines, speaker_mapping = self.process_deepgram_transcript(raw_text)
        else:
            raise ValueError(f"Unknown format_type: {format_type}. Use 'doctor_patient' or 'deepgram'")

        # Join lines into single string
        indexed_transcript = '\n'.join(indexed_lines)

        return indexed_transcript, speaker_mapping


# Convenience functions for standalone usage
def index_transcript_from_file(file_path: str, format_type: str = "doctor_patient") -> Tuple[str, Dict[str, str]]:
    """
    Convenience function to index a transcript file.

    Parameters
    ----------
    file_path : str
        Path to transcript file
    format_type : str, default "doctor_patient"
        Transcript format: "doctor_patient" or "deepgram"

    Returns
    -------
    Tuple[str, Dict[str, str]]
        Tuple of (indexed_transcript_string, speaker_mapping)

    Example
    -------
    >>> indexed_transcript, mapping = index_transcript_from_file("transcript.txt")
    >>> print(indexed_transcript[:100])
    '[T1] [SPEAKER 0]: How are you feeling today?
    [T2] [SPEAKER 1]: I'm feeling much better.'
    """
    indexer = TranscriptIndexer()
    return indexer.process_transcript_file(file_path, format_type)


def index_transcript_from_text(text: str, format_type: str = "doctor_patient") -> Tuple[str, Dict[str, str]]:
    """
    Convenience function to index transcript text.

    Parameters
    ----------
    text : str
        Raw transcript text
    format_type : str, default "doctor_patient"
        Transcript format: "doctor_patient" or "deepgram"

    Returns
    -------
    Tuple[str, Dict[str, str]]
        Tuple of (indexed_transcript_string, speaker_mapping)

    Example
    -------
    >>> text = "DOCTOR\\nHow are you?\\nPATIENT\\nI'm fine."
    >>> indexed_transcript, mapping = index_transcript_from_text(text)
    >>> print(mapping)
    {'DOCTOR': 'SPEAKER 0', 'PATIENT': 'SPEAKER 1'}
    """
    indexer = TranscriptIndexer()

    if format_type == "doctor_patient":
        indexed_lines, speaker_mapping = indexer.process_doctor_patient_transcript(text)
    elif format_type == "deepgram":
        indexed_lines, speaker_mapping = indexer.process_deepgram_transcript(text)
    else:
        raise ValueError(f"Unknown format_type: {format_type}. Use 'doctor_patient' or 'deepgram'")

    indexed_transcript = '\n'.join(indexed_lines)
    return indexed_transcript, speaker_mapping


if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    import os

    # Test with sample transcript
    transcript_path = r"C:\Users\Yijia Liu\Scribing Project\medical_transcripts\A074.txt"
    with open(transcript_path, 'r', encoding='utf-8') as f:
        sample_transcript = f.read()

    print("Testing TranscriptIndexer...")
    print("=" * 50)

    # Test with text directly
    indexed_transcript, speaker_mapping = index_transcript_from_text(sample_transcript)

    print("Original transcript:")
    print(sample_transcript)
    print("\nIndexed transcript:")
    print(indexed_transcript)
    print("\nSpeaker mapping:")
    print(speaker_mapping)