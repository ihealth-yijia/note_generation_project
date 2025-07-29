"""
Batch Transcript Processor Module

This module provides functionality to batch process medical transcripts through the complete pipeline:
1. Convert raw transcripts to indexed format
2. Generate SOAP notes using GPT
3. Create reverse lookup data structures
4. Save results to DataFrame

The processor handles 400+ transcript files and provides comprehensive error handling,
progress tracking, and checkpoint functionality.
"""

import os
import re
import json
import uuid
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass, asdict
from hashlib import md5
import time

# Import existing modules
import sys
from generate_soap_gpt import generate_soap_note


class TranscriptIndexer:
    """
    Convert raw medical transcripts to indexed format with speaker tags and turn numbers.
    """

    def __init__(self):
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
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading transcript file {file_path}: {e}")

    def process_doctor_patient_transcript(self, text: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Process transcript text in DOCTOR/PATIENT style and return indexed lines.

        Parameters
        ----------
        text : str
            Raw transcript text

        Returns
        -------
        Tuple[List[str], Dict[str, str]]
            Tuple of (indexed_lines, speaker_mapping)
        """
        speaker_map = {}
        speaker_counter = 0
        current_speaker = None
        t_counter = 1
        output_lines = []

        # Remove metadata lines
        cleaned_lines = [
            line.strip()
            for line in text.splitlines()
            if
            line.strip() and not re.match(r"^(Date|Time|DOCTOR OUT|PATIENT LEAVES|DD:|DT:|INFOPRO|TRANSACTION)", line)
        ]

        for line in cleaned_lines:
            # Match speaker lines like "DOCTOR B552" or "PATIENT"
            speaker_match = re.match(r"^(DOCTOR|PATIENT)(?:\s+\S+)?$", line)
            if speaker_match:
                speaker_label = speaker_match.group(1)
                if speaker_label not in speaker_map:
                    speaker_map[speaker_label] = f"SPEAKER {speaker_counter}"
                    speaker_counter += 1
                current_speaker = speaker_map[speaker_label]
            elif current_speaker:
                # Split line into sentences and add turn markers
                utterances = re.split(r'(?<=[.!?])\s+|\n', line)
                for utt in utterances:
                    if utt.strip():
                        output_lines.append(f"[T{t_counter}] [{current_speaker}]: {utt.strip()}")
                        t_counter += 1

        return output_lines, speaker_map

    def process_deepgram_transcript(self, text: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Process Deepgram-style transcript with Speaker X: format.

        Parameters
        ----------
        text : str
            Raw transcript text

        Returns
        -------
        Tuple[List[str], Dict[str, str]]
            Tuple of (indexed_lines, speaker_mapping)
        """
        speaker_blocks = re.split(r'\n(?=Speaker \d+:)', text.strip())
        numbered_lines = []
        line_counter = 1

        for block in speaker_blocks:
            match = re.match(r'Speaker (\d+):\s*(.*)', block, re.DOTALL)
            if not match:
                continue
            speaker_id = match.group(1)
            content = match.group(2)

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+|\n', content.strip())
            for sentence in sentences:
                if sentence.strip():
                    line = f"[T{line_counter}] [SPEAKER {speaker_id}]: {sentence.strip()}"
                    numbered_lines.append(line)
                    line_counter += 1

        # Create speaker mapping
        speaker_map = {f"SPEAKER {i}": f"SPEAKER {i}" for i in range(10)}  # Default mapping
        return numbered_lines, speaker_map


@dataclass
class ProcessingResult:
    """Data class to store processing results for a single transcript."""
    transcript_id: str
    raw_transcript: str
    indexed_transcript: str
    generated_soap_note: str
    normalized_soap_note: str
    reverse_lookup_data: Dict[str, Any]
    processing_status: str
    error_message: Optional[str]
    processing_time: str
    processing_duration: float


class ReverseLookupBuilder:
    """
    Build reverse lookup data structures for SOAP notes to transcript mapping.
    """

    def __init__(self):
        self.sentence_pattern = re.compile(r'\[T(\d+)(?:-T?(\d+))?\]\[SPEAKER (\d+)\]')

    def parse_transcript_lines(self, indexed_lines: List[str], speaker_mapping: Dict[str, str] = None) -> Dict[
        str, Dict[str, Any]]:
        """
        Parse indexed transcript lines into structured format.

        Parameters
        ----------
        indexed_lines : List[str]
            List of indexed transcript lines
        speaker_mapping : Dict[str, str], optional
            Mapping of speaker IDs to roles

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Transcript mapping dictionary
        """
        speaker_mapping = speaker_mapping or {}
        transcript_map = {}

        line_pattern = re.compile(r'^\[T(\d+)\]\s+\[(SPEAKER \d+)\]:\s*(.*)$')

        for line in indexed_lines:
            match = line_pattern.match(line.strip())
            if not match:
                continue

            t_idx, speaker_id, text = match.groups()
            key = f"T{t_idx}"

            transcript_map[key] = {
                "idx": int(t_idx),
                "speaker_id": speaker_id,
                "role": speaker_mapping.get(speaker_id, "unknown"),
                "text": text.strip()
            }

        return transcript_map

    def extract_soap_sentences(self, soap_note: str) -> List[Dict[str, Any]]:
        """
        Extract sentences from SOAP note with their references.

        Parameters
        ----------
        soap_note : str
            Generated SOAP note text

        Returns
        -------
        List[Dict[str, Any]]
            List of sentence dictionaries
        """
        sentences = []
        current_section = None

        for line in soap_note.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            if line.endswith(':') and line.split(':')[0] in ['Subjective', 'Objective', 'Assessment', 'Plan']:
                current_section = line.split(':')[0]
                continue

            # Extract references from the line
            references = self.sentence_pattern.findall(line)
            if references and current_section:
                # Remove reference tags from text
                clean_text = re.sub(r'\s*\[T[^\]]+\]\[SPEAKER [^\]]+\]\s*$', '', line).strip()

                sentence_data = {
                    "id": str(uuid.uuid4()),
                    "section": current_section,
                    "text": clean_text,
                    "raw_text": line,
                    "references": references
                }
                sentences.append(sentence_data)

        return sentences

    def build_reverse_lookup_data(self, soap_note: str, indexed_lines: List[str],
                                  speaker_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Build comprehensive reverse lookup data structure.

        Parameters
        ----------
        soap_note : str
            Generated SOAP note
        indexed_lines : List[str]
            Indexed transcript lines
        speaker_mapping : Dict[str, str], optional
            Speaker role mapping

        Returns
        -------
        Dict[str, Any]
            Complete reverse lookup data structure
        """
        transcript_map = self.parse_transcript_lines(indexed_lines, speaker_mapping)
        soap_sentences = self.extract_soap_sentences(soap_note)

        # Build sentence to transcript mapping
        sentence_to_transcript = {}

        for sentence in soap_sentences:
            transcript_refs = []

            for ref in sentence['references']:
                start_t, end_t, speaker = ref[0], ref[1] or ref[0], ref[2]

                # Expand T-ranges
                start_num = int(start_t)
                end_num = int(end_t)

                for t_num in range(start_num, end_num + 1):
                    t_key = f"T{t_num}"
                    if t_key in transcript_map:
                        transcript_refs.append({
                            "t": t_key,
                            "speaker_id": transcript_map[t_key]["speaker_id"],
                            "role": transcript_map[t_key]["role"],
                            "text": transcript_map[t_key]["text"]
                        })

            sentence_to_transcript[sentence['id']] = transcript_refs

        return {
            "transcript_map": transcript_map,
            "soap_sentences": soap_sentences,
            "sentence_to_transcript": sentence_to_transcript,
            "speaker_mapping": speaker_mapping or {}
        }


class BatchTranscriptProcessor:
    """
    Main class for batch processing medical transcripts through the complete pipeline.
    """

    def __init__(self, api_key: str, prompt_template_path: str, transcript_dir: str,
                 output_dir: str, model_name: str = "chatgpt-4o-latest"):
        """
        Initialize the batch processor.

        Parameters
        ----------
        api_key : str
            OpenAI API key
        prompt_template_path : str
            Path to SOAP prompt template file
        transcript_dir : str
            Directory containing transcript files
        output_dir : str
            Directory for output files
        model_name : str, default "chatgpt-4o-latest"
            OpenAI model to use
        """
        self.api_key = api_key
        self.prompt_template_path = Path(prompt_template_path)
        self.transcript_dir = Path(transcript_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        # Initialize components
        self.indexer = TranscriptIndexer()
        self.reverse_lookup_builder = ReverseLookupBuilder()

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        # Setup logging
        self._setup_logging()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_prompt_template(self) -> str:
        """Load SOAP prompt template from file."""
        try:
            # Check for soap_prompt.txt first, then other common names
            possible_names = ['problem_based_note_prompt.txt','soap_prompt.txt', 'prompt.txt', 'soap_template.txt']

            for name in possible_names:
                template_file = self.prompt_template_path / name
                if template_file.exists():
                    return template_file.read_text(encoding='utf-8')

            # If no specific file found, use the first .txt file
            txt_files = list(self.prompt_template_path.glob('*.txt'))
            if txt_files:
                return txt_files[0].read_text(encoding='utf-8')

            raise FileNotFoundError(f"No prompt template found in {self.prompt_template_path}")

        except Exception as e:
            raise Exception(f"Error loading prompt template: {e}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_transcript_files(self) -> List[Path]:
        """
        Get list of transcript files to process.

        Returns
        -------
        List[Path]
            List of transcript file paths
        """
        # Look for .txt files (most common)
        transcript_files = list(self.transcript_dir.glob('*.txt'))

        if not transcript_files:
            # Also check for other common formats
            for pattern in ['*.docx', '*.doc', '*.rtf']:
                transcript_files.extend(list(self.transcript_dir.glob(pattern)))

        self.logger.info(f"Found {len(transcript_files)} transcript files")
        return sorted(transcript_files)

    def process_single_transcript(self, file_path: Path) -> ProcessingResult:
        """
        Process a single transcript through the complete pipeline.

        Parameters
        ----------
        file_path : Path
            Path to transcript file

        Returns
        -------
        ProcessingResult
            Processing result data
        """
        start_time = time.time()
        transcript_id = file_path.stem

        try:
            # Step 1: Read raw transcript
            self.logger.info(f"Processing {transcript_id}: Reading raw transcript")
            raw_transcript = self.indexer.read_transcript(str(file_path))

            # Step 2: Create indexed transcript
            self.logger.info(f"Processing {transcript_id}: Creating indexed transcript")
            try:
                # Try doctor/patient format first
                indexed_lines, speaker_mapping = self.indexer.process_doctor_patient_transcript(raw_transcript)
            except:
                # Fallback to Deepgram format
                indexed_lines, speaker_mapping = self.indexer.process_deepgram_transcript(raw_transcript)

            indexed_transcript = '\n'.join(indexed_lines)

            # Step 3: Generate SOAP note
            self.logger.info(f"Processing {transcript_id}: Generating SOAP note")
            generated_soap_note = generate_soap_note(
                prompt_template=self.prompt_template,
                indexed_transcript=indexed_transcript,
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2,
                max_tokens=3000
            )

            # Step 4: Normalize SOAP note (already done by generate_soap_note)
            normalized_soap_note = generated_soap_note

            # Step 5: Build reverse lookup data
            self.logger.info(f"Processing {transcript_id}: Building reverse lookup data")
            reverse_lookup_data = self.reverse_lookup_builder.build_reverse_lookup_data(
                generated_soap_note, indexed_lines, speaker_mapping
            )

            processing_duration = time.time() - start_time

            return ProcessingResult(
                transcript_id=transcript_id,
                raw_transcript=raw_transcript,
                indexed_transcript=indexed_transcript,
                generated_soap_note=generated_soap_note,
                normalized_soap_note=normalized_soap_note,
                reverse_lookup_data=reverse_lookup_data,
                processing_status="success",
                error_message=None,
                processing_time=datetime.now().isoformat(),
                processing_duration=processing_duration
            )

        except Exception as e:
            processing_duration = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Error processing {transcript_id}: {error_msg}")

            return ProcessingResult(
                transcript_id=transcript_id,
                raw_transcript=raw_transcript if 'raw_transcript' in locals() else "",
                indexed_transcript="",
                generated_soap_note="",
                normalized_soap_note="",
                reverse_lookup_data={},
                processing_status="error",
                error_message=error_msg,
                processing_time=datetime.now().isoformat(),
                processing_duration=processing_duration
            )

    def save_checkpoint(self, results: List[ProcessingResult], checkpoint_name: str = "checkpoint"):
        """
        Save intermediate results as checkpoint.

        Parameters
        ----------
        results : List[ProcessingResult]
            Processing results to save
        checkpoint_name : str, default "checkpoint"
            Name for checkpoint file
        """
        checkpoint_file = self.output_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Convert reverse_lookup_data to JSON string for storage
            result_dict['reverse_lookup_data'] = json.dumps(result_dict['reverse_lookup_data'])
            serializable_results.append(result_dict)

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, checkpoint_file: str) -> List[ProcessingResult]:
        """
        Load results from checkpoint file.

        Parameters
        ----------
        checkpoint_file : str
            Path to checkpoint file

        Returns
        -------
        List[ProcessingResult]
            Loaded processing results
        """
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for item in data:
            # Convert reverse_lookup_data back from JSON string
            item['reverse_lookup_data'] = json.loads(item['reverse_lookup_data'])
            results.append(ProcessingResult(**item))

        return results

    def results_to_dataframe(self, results: List[ProcessingResult]) -> pd.DataFrame:
        """
        Convert processing results to pandas DataFrame.

        Parameters
        ----------
        results : List[ProcessingResult]
            Processing results

        Returns
        -------
        pd.DataFrame
            Results DataFrame
        """
        data = []
        for result in results:
            row = {
                'transcript_id': result.transcript_id,
                'raw_transcript': result.raw_transcript,
                'indexed_transcript': result.indexed_transcript,
                'generated_soap_note': result.generated_soap_note,
                'normalized_soap_note': result.normalized_soap_note,
                'reverse_lookup_data': json.dumps(result.reverse_lookup_data),  # Store as JSON string
                'processing_status': result.processing_status,
                'error_message': result.error_message,
                'processing_time': result.processing_time,
                'processing_duration': result.processing_duration
            }
            data.append(row)

        return pd.DataFrame(data)

    def process_all_transcripts(self, save_checkpoints: bool = True, checkpoint_interval: int = 10) -> pd.DataFrame:
        """
        Process all transcripts in the directory.

        Parameters
        ----------
        save_checkpoints : bool, default True
            Whether to save periodic checkpoints
        checkpoint_interval : int, default 10
            Save checkpoint every N transcripts

        Returns
        -------
        pd.DataFrame
            Complete results DataFrame
        """
        transcript_files = self.get_transcript_files()
        results = []

        self.logger.info(f"Starting batch processing of {len(transcript_files)} transcripts")

        for i, file_path in enumerate(tqdm(transcript_files, desc="Processing transcripts")):
            result = self.process_single_transcript(file_path)
            results.append(result)

            # Save checkpoint periodically
            if save_checkpoints and (i + 1) % checkpoint_interval == 0:
                self.save_checkpoint(results, f"checkpoint_batch_{i + 1}")

        # Save final results
        df = self.results_to_dataframe(results)

        # Save to multiple formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # CSV format
        csv_file = self.output_dir / f"batch_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')

        # Parquet format (more efficient for large datasets)
        parquet_file = self.output_dir / f"batch_results_{timestamp}.parquet"
        df.to_parquet(parquet_file, index=False)

        # Excel format (for easy viewing)
        excel_file = self.output_dir / f"batch_results_{timestamp}.xlsx"
        df.to_excel(excel_file, index=False)

        # Summary statistics
        success_count = len(df[df['processing_status'] == 'success'])
        error_count = len(df[df['processing_status'] == 'error'])
        avg_duration = df['processing_duration'].mean()

        self.logger.info(f"Batch processing completed:")
        self.logger.info(f"  Total files: {len(transcript_files)}")
        self.logger.info(f"  Successful: {success_count}")
        self.logger.info(f"  Errors: {error_count}")
        self.logger.info(f"  Average processing time: {avg_duration:.2f} seconds")
        self.logger.info(f"  Results saved to: {csv_file}")

        return df


def main():
    """
    Example usage of the batch processor.
    """
    # Configuration
    api_key = "your-openai-api-key"  # Set your API key here
    prompt_template_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\prompt_templates"
    transcript_dir = r"C:\Users\Yijia Liu\Scribing Project\medical_transcripts"
    output_dir = r"C:\Users\Yijia Liu\Scribing Project\batch_processing_results"

    # Create processor
    processor = BatchTranscriptProcessor(
        api_key=api_key,
        prompt_template_path=prompt_template_path,
        transcript_dir=transcript_dir,
        output_dir=output_dir
    )

    # Process all transcripts
    results_df = processor.process_all_transcripts(
        save_checkpoints=True,
        checkpoint_interval=5  # Save checkpoint every 5 files
    )

    print(f"Processing completed. Results shape: {results_df.shape}")
    print(f"Success rate: {(results_df['processing_status'] == 'success').mean():.2%}")


if __name__ == "__main__":
    main()