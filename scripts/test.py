"""
Complete SOAP Note Generation Pipeline

This module provides a complete three-stage pipeline for medical transcript processing:
1. Raw transcript → Indexed transcript (with speaker tags and turn numbers)
2. Indexed transcript + Template1 → Stage 1 SOAP note
3. Stage 1 note + Template2 → Final formatted SOAP note

The pipeline supports batch processing with error handling, progress tracking, and checkpoints.
"""

import os
import re
import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass, asdict
import time

# Import existing modules
from generate_soap_gpt import generate_soap_note
from soap_note_formatter import format_soap_note
from transcript_indexer import TranscriptIndexer
from soap_note_parser import SOAPNoteParser


@dataclass
class PipelineResult:
    """Data class to store complete pipeline results for a single transcript."""
    transcript_id: str
    raw_transcript: str
    indexed_transcript: str
    stage1_soap_note: str
    final_soap_note: str
    parsed_soap_json: Dict[str, Any]  # New field for parsed JSON
    speaker_mapping: Dict[str, str]
    processing_status: str
    error_message: Optional[str]
    processing_time: str
    stage1_duration: float
    stage2_duration: float
    total_duration: float


class CompleteSoapPipeline:
    """
    Complete three-stage SOAP note generation pipeline.
    """

    def __init__(self, api_key: str, stage1_template_path: str, stage2_template_path: str,
                 transcript_dir: str, output_dir: str, model_name: str = "chatgpt-4o-latest"):
        """
        Initialize the complete pipeline.

        Parameters
        ----------
        api_key : str
            OpenAI API key
        stage1_template_path : str
            Path to stage 1 prompt template file
        stage2_template_path : str
            Path to stage 2 prompt template file
        transcript_dir : str
            Directory containing transcript files
        output_dir : str
            Directory for output files
        model_name : str, default "chatgpt-4o-latest"
            OpenAI model to use
        """
        self.api_key = api_key
        self.stage1_template_path = Path(stage1_template_path)
        self.stage2_template_path = Path(stage2_template_path)
        self.transcript_dir = Path(transcript_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        # Initialize components
        self.indexer = TranscriptIndexer()
        self.parser = SOAPNoteParser()  # Add parser

        # Create output directory first
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load prompt templates
        self.stage1_template = self._load_template(self.stage1_template_path)
        self.stage2_template = self._load_template(self.stage2_template_path)

        # Setup logging
        self._setup_logging()



    def _load_template(self, template_path: Path) -> str:
        """Load prompt template from file."""
        try:
            if template_path.is_file():
                return template_path.read_text(encoding='utf-8')
            else:
                raise FileNotFoundError(f"Template file not found: {template_path}")
        except Exception as e:
            raise Exception(f"Error loading template {template_path}: {e}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"pipeline_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

    def process_single_transcript(self, file_path: Path, detail_level: str = "standard",
                                  organization_style: str = "bullet_point") -> PipelineResult:
        """
        Process a single transcript through the complete three-stage pipeline.

        Parameters
        ----------
        file_path : Path
            Path to transcript file
        detail_level : str, default "standard"
            Detail level for final formatting: "brief", "standard", "detailed", "comprehensive"
        organization_style : str, default "bullet_point"
            Organization style for final formatting: "paragraph", "bullet_point"

        Returns
        -------
        PipelineResult
            Complete pipeline result data
        """
        start_time = time.time()
        transcript_id = file_path.stem
        raw_transcript = ""  # Initialize here to avoid UnboundLocalError

        try:
            # Stage 1: Read and index raw transcript
            self.logger.info(f"Processing {transcript_id}: Stage 1 - Reading and indexing transcript")
            raw_transcript = self.indexer.read_transcript(str(file_path))
            indexed_transcript, speaker_mapping = self.indexer.process_transcript_file(str(file_path), "doctor_patient")

            # Stage 2: Generate initial SOAP note
            self.logger.info(f"Processing {transcript_id}: Stage 2 - Generating initial SOAP note")
            stage1_start = time.time()

            stage1_soap_note = generate_soap_note(
                prompt_template=self.stage1_template,
                indexed_transcript=indexed_transcript,
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2,
                max_tokens=3000
            )

            stage1_duration = time.time() - stage1_start

            # Stage 3: Format final SOAP note
            self.logger.info(f"Processing {transcript_id}: Stage 3 - Formatting final SOAP note")
            stage2_start = time.time()

            final_soap_note = format_soap_note(
                prompt_template=self.stage2_template,
                current_note=stage1_soap_note,
                detail_level=detail_level,
                organization_style=organization_style,
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2,
                max_tokens=5000
            )

            stage2_duration = time.time() - stage2_start

            # Stage 4: Parse final SOAP note into JSON format
            self.logger.info(f"Processing {transcript_id}: Stage 4 - Parsing SOAP note to JSON")
            parsed_soap_json = self.parser.parse_soap_note(final_soap_note, detail_level, organization_style)

            total_duration = time.time() - start_time

            return PipelineResult(
                transcript_id=transcript_id,
                raw_transcript=raw_transcript,
                indexed_transcript=indexed_transcript,
                stage1_soap_note=stage1_soap_note,
                final_soap_note=final_soap_note,
                parsed_soap_json=parsed_soap_json,
                speaker_mapping=speaker_mapping,
                processing_status="success",
                error_message=None,
                processing_time=datetime.now().isoformat(),
                stage1_duration=stage1_duration,
                stage2_duration=stage2_duration,
                total_duration=total_duration
            )

        except Exception as e:
            total_duration = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Error processing {transcript_id}: {error_msg}")

            # Ensure raw_transcript is available even if early failure occurs
            if not raw_transcript:
                try:
                    raw_transcript = self.indexer.read_transcript(str(file_path))
                except:
                    raw_transcript = ""

            return PipelineResult(
                transcript_id=transcript_id,
                raw_transcript=raw_transcript,
                indexed_transcript=indexed_transcript if 'indexed_transcript' in locals() else "",
                stage1_soap_note=stage1_soap_note if 'stage1_soap_note' in locals() else "",
                final_soap_note="",
                parsed_soap_json={},  # Empty dict for failed cases
                speaker_mapping=speaker_mapping if 'speaker_mapping' in locals() else {},
                processing_status="error",
                error_message=error_msg,
                processing_time=datetime.now().isoformat(),
                stage1_duration=stage1_duration if 'stage1_duration' in locals() else 0.0,
                stage2_duration=0.0,
                total_duration=total_duration
            )

    def save_checkpoint(self, results: List[PipelineResult], checkpoint_name: str = "checkpoint"):
        """
        Save intermediate results as checkpoint.

        Parameters
        ----------
        results : List[PipelineResult]
            Processing results to save
        checkpoint_name : str, default "checkpoint"
            Name for checkpoint file
        """
        checkpoint_file = self.output_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            serializable_results.append(result_dict)

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, checkpoint_file: str) -> List[PipelineResult]:
        """
        Load results from checkpoint file.

        Parameters
        ----------
        checkpoint_file : str
            Path to checkpoint file

        Returns
        -------
        List[PipelineResult]
            Loaded processing results
        """
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for item in data:
            results.append(PipelineResult(**item))

        return results

    def results_to_dataframe(self, results: List[PipelineResult]) -> pd.DataFrame:
        """
        Convert processing results to pandas DataFrame.

        Parameters
        ----------
        results : List[PipelineResult]
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
                'stage1_soap_note': result.stage1_soap_note,
                'final_soap_note': result.final_soap_note,
                'parsed_soap_json': json.dumps(result.parsed_soap_json),  # Store as JSON string
                'speaker_mapping': json.dumps(result.speaker_mapping),
                'processing_status': result.processing_status,
                'error_message': result.error_message,
                'processing_time': result.processing_time,
                'stage1_duration': result.stage1_duration,
                'stage2_duration': result.stage2_duration,
                'total_duration': result.total_duration
            }
            data.append(row)

        return pd.DataFrame(data)

    def process_all_transcripts(self, detail_level: str = "standard",
                                organization_style: str = "bullet_point",
                                save_checkpoints: bool = True,
                                checkpoint_interval: int = 10) -> pd.DataFrame:
        """
        Process all transcripts through the complete pipeline.

        Parameters
        ----------
        detail_level : str, default "standard"
            Detail level for final formatting
        organization_style : str, default "bullet_point"
            Organization style for final formatting
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

        self.logger.info(f"Starting complete pipeline processing of {len(transcript_files)} transcripts")
        self.logger.info(f"Detail level: {detail_level}, Organization style: {organization_style}")

        for i, file_path in enumerate(tqdm(transcript_files, desc="Processing transcripts")):
            result = self.process_single_transcript(file_path, detail_level, organization_style)
            results.append(result)

            # Save checkpoint periodically
            if save_checkpoints and (i + 1) % checkpoint_interval == 0:
                self.save_checkpoint(results, f"checkpoint_pipeline_{i + 1}")

        # Save final results
        df = self.results_to_dataframe(results)

        # Save individual JSON files for each transcript
        json_dir = self.output_dir / "parsed_json_notes"
        json_dir.mkdir(exist_ok=True)

        for result in results:
            if result.processing_status == "success" and result.parsed_soap_json:
                json_file = json_dir / f"{result.transcript_id}_parsed.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result.parsed_soap_json, f, indent=2, ensure_ascii=False)

        # Save to multiple formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # CSV format
        csv_file = self.output_dir / f"complete_pipeline_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')

        # Parquet format (more efficient for large datasets)
        parquet_file = self.output_dir / f"complete_pipeline_results_{timestamp}.parquet"
        df.to_parquet(parquet_file, index=False)

        # Excel format (for easy viewing)
        excel_file = self.output_dir / f"complete_pipeline_results_{timestamp}.xlsx"
        df.to_excel(excel_file, index=False)

        # Summary statistics
        success_count = len(df[df['processing_status'] == 'success'])
        error_count = len(df[df['processing_status'] == 'error'])
        avg_total_duration = df['total_duration'].mean()
        avg_stage1_duration = df['stage1_duration'].mean()
        avg_stage2_duration = df['stage2_duration'].mean()

        self.logger.info(f"Complete pipeline processing finished:")
        self.logger.info(f"  Total files: {len(transcript_files)}")
        self.logger.info(f"  Successful: {success_count}")
        self.logger.info(f"  Errors: {error_count}")
        self.logger.info(f"  Average total time: {avg_total_duration:.2f} seconds")
        self.logger.info(f"  Average stage 1 time: {avg_stage1_duration:.2f} seconds")
        self.logger.info(f"  Average stage 2 time: {avg_stage2_duration:.2f} seconds")
        self.logger.info(f"  Results saved to: {csv_file}")

        return df

    def process_single_file_demo(self, file_path: str, detail_level: str = "standard",
                                 organization_style: str = "bullet_point"):
        """
        Demo function to process a single file and display results.

        Parameters
        ----------
        file_path : str
            Path to single transcript file
        detail_level : str, default "standard"
            Detail level for formatting
        organization_style : str, default "bullet_point"
            Organization style for formatting
        """
        file_path = Path(file_path)

        print(f"Processing single file: {file_path.name}")
        print("=" * 60)

        result = self.process_single_transcript(file_path, detail_level, organization_style)

        if result.processing_status == "success":
            print(f"✓ Successfully processed {result.transcript_id}")
            print(f"  Stage 1 duration: {result.stage1_duration:.2f} seconds")
            print(f"  Stage 2 duration: {result.stage2_duration:.2f} seconds")
            print(f"  Total duration: {result.total_duration:.2f} seconds")

            print("\n" + "=" * 60)
            print("FINAL SOAP NOTE (RAW):")
            print("=" * 60)
            print(result.final_soap_note)

            print("\n" + "=" * 60)
            print("PARSED JSON STRUCTURE:")
            print("=" * 60)
            print("Subjective items:")
            for i, item in enumerate(result.parsed_soap_json["parsed_sections"]["subjective"][:3], 1):
                print(f"  {i}. {item[:80]}...")

            print(f"\nObjective items: {len(result.parsed_soap_json['parsed_sections']['objective'])}")
            print(f"Assessment items: {len(result.parsed_soap_json['parsed_sections']['assessment'])}")
            print(f"Plan items: {len(result.parsed_soap_json['parsed_sections']['plan'])}")
            print(f"Speaker mapping: {result.parsed_soap_json['parsed_sections']['speaker_role_mapping']}")

        else:
            print(f"✗ Error processing {result.transcript_id}: {result.error_message}")


def main():
    """
    Example usage of the complete pipeline.
    """
    # Configuration - Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Template paths
    stage1_template_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\prompt_templates\soap_prompt.txt"
    stage2_template_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\prompt_templates\soap_step2_generation_prompt.txt"

    # Directories
    transcript_dir = r"C:\Users\Yijia Liu\Scribing Project\medical_transcripts"
    output_dir = r"C:\Users\Yijia Liu\Scribing Project\complete_pipeline_results"

    # Create pipeline
    pipeline = CompleteSoapPipeline(
        api_key=api_key,
        stage1_template_path=stage1_template_path,
        stage2_template_path=stage2_template_path,
        transcript_dir=transcript_dir,
        output_dir=output_dir
    )

    # Option 1: Process single file (for testing)
    single_file = r"C:\Users\Yijia Liu\Scribing Project\medical_transcripts\A074.txt"
    pipeline.process_single_file_demo(single_file)  # Using defaults: standard + bullet_point

    # Option 2: Process all transcripts
    # results_df = pipeline.process_all_transcripts()  # Using defaults: standard + bullet_point
    # print(f"Pipeline completed. Results shape: {results_df.shape}")
    # print(f"Success rate: {(results_df['processing_status'] == 'success').mean():.2%}")


if __name__ == "__main__":
    main()