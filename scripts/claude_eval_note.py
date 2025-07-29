import pandas as pd
import boto3
import json
import time
import os
import datetime
from botocore.config import Config


# from generate_soap import indexed_transcript


# === 1. Load system prompt template ===
def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template text from a local file"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# === 2. Render prompt with transcript and note ===
def render_prompt(prompt_template: str, indexed_transcript: str, normalized_note: str) -> str:
    """Insert actual transcript and note into the prompt template"""
    try:
        # First, let's check what placeholders are in the template
        import re
        placeholders = re.findall(r'\{([^}]+)\}', prompt_template)
        print(f"📝 Found placeholders in template: {placeholders}")

        # Try to format the prompt
        formatted_prompt = prompt_template.format(
            indexed_transcript=indexed_transcript,
            normalized_note=normalized_note
        )

        return formatted_prompt

    except KeyError as e:
        print(f"❌ KeyError in prompt template: {e}")
        print(f"❌ Available placeholders should be: {{indexed_transcript}} and {{normalized_note}}")
        print(f"❌ Check your prompt template file for correct placeholder names")
        raise
    except Exception as e:
        print(f"❌ Error formatting prompt: {e}")
        print(f"❌ Prompt template preview (first 200 chars):")
        print(f"   {prompt_template[:200]}...")
        raise


# === 3. Call Claude Sonnet 4 via AWS Bedrock ===
def call_claude_sonnet_converse(
        rendered_prompt: str,
        profile_arn: str,
        profile_name: str = 'ihealth-test',
        region_name: str = 'us-west-2'
) -> str:
    """
    Call Claude Sonnet 4 via AWS Bedrock using Converse API + Inference Profile
    """
    try:
        session = boto3.Session(profile_name=profile_name)

        # Configure the client with longer timeouts
        config = Config(
            read_timeout=300,  # 5 minutes
            connect_timeout=60,  # 1 minute
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )

        bedrock_runtime = session.client(
            'bedrock-runtime',
            region_name=region_name,
            config=config
        )

        response = bedrock_runtime.converse(
            modelId=profile_arn,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": rendered_prompt}]
                }
            ],
            inferenceConfig={
                "temperature": 0.5,
                "topP": 1.0,
                "maxTokens": 5000
            }
        )

        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        if "Read timeout" in str(e) or "timeout" in str(e).lower():
            raise TimeoutError(f"AWS Bedrock timeout: {e}")
        else:
            raise Exception(f"Claude API call failed: {e}")


# === 4. Batch evaluation loop ===
def batch_evaluate_claude(df: pd.DataFrame, prompt_template: str, profile_arn: str) -> pd.DataFrame:
    """Iterate through each row of the DataFrame and evaluate using Claude Sonnet 4"""
    results = []
    total_rows = len(df)

    for i, row in df.iterrows():
        print(f"🔄 Processing row {i + 1}/{total_rows}...")

        # Safely get data and handle potential NaN values
        indexed_transcript = str(row.get('indexed_transcript', '')) if pd.notna(row.get('indexed_transcript')) else ''
        normalized_note = str(row.get('normalized_soap_note', '')) if pd.notna(row.get('normalized_soap_note')) else ''

        # Check if required data exists
        if not indexed_transcript.strip() or not normalized_note.strip():
            print(f"⚠️ Row {i + 1} missing required data")
            results.append({
                "raw_response": None,
                "error": "Missing indexed_transcript or normalized_soap_note",
                "transcript_id": row.get('transcript_id', ''),
                "processing_status": "failed"
            })
            continue

        try:
            # Generate the full prompt and send to Claude
            full_prompt = render_prompt(prompt_template, indexed_transcript, normalized_note)
            print(f"📝 Sending prompt for row {i + 1}... (prompt length: {len(full_prompt)} chars)")

            # Add timestamp for tracking
            start_time = datetime.datetime.now()

            # Call Claude Sonnet 4
            response_text = call_claude_sonnet_converse(full_prompt, profile_arn)

            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"🕐 API call completed in {duration:.1f} seconds")

            # Check if response is empty
            if not response_text or not response_text.strip():
                print(f"⚠️ Row {i + 1} returned empty response")
                results.append({
                    "raw_response": "",
                    "error": "Empty response",
                    "transcript_id": row.get('transcript_id', ''),
                    "processing_status": "failed"
                })
                continue

            # Try parsing the JSON result; fallback to raw text if malformed
            try:
                # Clean potential markdown formatting
                cleaned_response = response_text.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()

                parsed_json = json.loads(cleaned_response)
                # Add metadata for tracking
                parsed_json['raw_response'] = response_text
                parsed_json['transcript_id'] = row.get('transcript_id', '')
                parsed_json['processing_status'] = "completed"
                results.append(parsed_json)
                print(f"✅ Row {i + 1} processed successfully")

            except json.JSONDecodeError as je:
                print(f"⚠️ Row {i + 1} returned malformed JSON: {str(je)}")
                results.append({
                    "raw_response": response_text,
                    "error": f"JSONDecodeError: {str(je)}",
                    "transcript_id": row.get('transcript_id', ''),
                    "processing_status": "json_error"
                })

        except TimeoutError as te:
            print(f"⏰ Row {i + 1} timed out: {te}")
            results.append({
                "raw_response": None,
                "error": f"Timeout: {str(te)}",
                "transcript_id": row.get('transcript_id', ''),
                "processing_status": "timeout_error"
            })
        except Exception as e:
            print(f"❌ Error at row {i + 1}: {e}")
            results.append({
                "raw_response": None,
                "error": str(e),
                "transcript_id": row.get('transcript_id', ''),
                "processing_status": "api_error"
            })

        # Add delay to avoid rate limiting and show progress
        if i < total_rows - 1:  # No need to wait after the last row
            print(f"⏳ Waiting 2 seconds before next request...")
            time.sleep(2)  # Increased to 2 seconds to avoid rate limiting

    # Add results to DataFrame
    df = df.copy()  # Avoid modifying the original DataFrame
    df['evaluation_claude_sonnet4'] = results

    # Print summary with detailed status breakdown
    successful_evaluations = sum(1 for r in results if r.get('processing_status') == 'completed')
    failed_evaluations = sum(1 for r in results if r.get('processing_status') == 'failed')
    json_errors = sum(1 for r in results if r.get('processing_status') == 'json_error')
    api_errors = sum(1 for r in results if r.get('processing_status') == 'api_error')
    timeout_errors = sum(1 for r in results if r.get('processing_status') == 'timeout_error')

    print(f"\n📊 Batch evaluation completed:")
    print(f"   Total rows: {total_rows}")
    print(f"   Successful: {successful_evaluations}")
    print(f"   Failed (missing data): {failed_evaluations}")
    print(f"   JSON parsing errors: {json_errors}")
    print(f"   API errors: {api_errors}")
    print(f"   Timeout errors: {timeout_errors}")

    return df


# === Helper function: Save intermediate results ===
def batch_evaluate_claude_with_checkpoint(df: pd.DataFrame, prompt_template: str, profile_arn: str,
                                          checkpoint_file: str = None) -> pd.DataFrame:
    """Batch evaluation with checkpoint functionality to resume from interruption"""
    results = []
    total_rows = len(df)

    # Load existing results if checkpoint file exists
    start_idx = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            checkpoint_df = pd.read_pickle(checkpoint_file)
            if len(checkpoint_df) > 0 and 'evaluation_claude_sonnet4' in checkpoint_df.columns:
                start_idx = len([r for r in checkpoint_df['evaluation_claude_sonnet4'] if r is not None])
                results = checkpoint_df['evaluation_claude_sonnet4'].tolist()[:start_idx]
                print(f"📂 Resuming from checkpoint: starting at row {start_idx + 1}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")

    for i in range(start_idx, total_rows):
        row = df.iloc[i]
        print(f"🔄 Processing row {i + 1}/{total_rows}...")

        # Safely get data
        indexed_transcript = str(row.get('indexed_transcript', '')) if pd.notna(row.get('indexed_transcript')) else ''
        normalized_note = str(row.get('normalized_soap_note', '')) if pd.notna(row.get('normalized_soap_note')) else ''

        if not indexed_transcript.strip() or not normalized_note.strip():
            print(f"⚠️ Row {i + 1} missing required data")
            results.append({
                "raw_response": None,
                "error": "Missing indexed_transcript or normalized_soap_note",
                "transcript_id": row.get('transcript_id', ''),
                "processing_status": "failed"
            })
            continue

        try:
            full_prompt = render_prompt(prompt_template, indexed_transcript, normalized_note)
            print(f"📝 Sending prompt for row {i + 1}... (prompt length: {len(full_prompt)} chars)")

            # Add timestamp for tracking
            start_time = datetime.datetime.now()

            # Call Claude Sonnet 4
            response_text = call_claude_sonnet_converse(full_prompt, profile_arn)

            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"🕐 API call completed in {duration:.1f} seconds")

            if not response_text or not response_text.strip():
                print(f"⚠️ Row {i + 1} returned empty response")
                results.append({
                    "raw_response": "",
                    "error": "Empty response",
                    "transcript_id": row.get('transcript_id', ''),
                    "processing_status": "failed"
                })
                continue

            try:
                cleaned_response = response_text.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()

                parsed_json = json.loads(cleaned_response)
                parsed_json['raw_response'] = response_text
                parsed_json['transcript_id'] = row.get('transcript_id', '')
                parsed_json['processing_status'] = "completed"
                results.append(parsed_json)
                print(f"✅ Row {i + 1} processed successfully")

            except json.JSONDecodeError as je:
                print(f"⚠️ Row {i + 1} returned malformed JSON: {str(je)}")
                results.append({
                    "raw_response": response_text,
                    "error": f"JSONDecodeError: {str(je)}",
                    "transcript_id": row.get('transcript_id', ''),
                    "processing_status": "json_error"
                })

        except TimeoutError as te:
            print(f"⏰ Row {i + 1} timed out: {te}")
            results.append({
                "raw_response": None,
                "error": f"Timeout: {str(te)}",
                "transcript_id": row.get('transcript_id', ''),
                "processing_status": "timeout_error"
            })
        except Exception as e:
            print(f"❌ Error at row {i + 1}: {e}")
            results.append({
                "raw_response": None,
                "error": str(e),
                "transcript_id": row.get('transcript_id', ''),
                "processing_status": "api_error"
            })

        # Save checkpoint every 10 rows
        if checkpoint_file and (i + 1) % 10 == 0:
            temp_df = df.iloc[:i + 1].copy()
            temp_df['evaluation_claude_sonnet4'] = results

            # Save checkpoint (pickle format for reliability)
            temp_df.to_pickle(checkpoint_file)

            # Also save progress CSV
            csv_checkpoint = checkpoint_file.replace('.pkl', '_progress.csv')
            temp_df.to_csv(csv_checkpoint, index=False)

            print(f"💾 Checkpoint saved at row {i + 1} (both pickle and CSV)")

        # Delay
        if i < total_rows - 1:
            time.sleep(2)

    # Add results to DataFrame
    df = df.copy()
    df['evaluation_claude_sonnet4'] = results

    # Save final checkpoint
    if checkpoint_file:
        df.to_pickle(checkpoint_file)
        # Save final CSV
        csv_checkpoint = checkpoint_file.replace('.pkl', '_final.csv')
        df.to_csv(csv_checkpoint, index=False)
        print(f"💾 Final results saved to {checkpoint_file} and {csv_checkpoint}")

    # Print summary with detailed status breakdown
    successful_evaluations = sum(1 for r in results if r.get('processing_status') == 'completed')
    failed_evaluations = sum(1 for r in results if r.get('processing_status') == 'failed')
    json_errors = sum(1 for r in results if r.get('processing_status') == 'json_error')
    api_errors = sum(1 for r in results if r.get('processing_status') == 'api_error')
    timeout_errors = sum(1 for r in results if r.get('processing_status') == 'timeout_error')

    print(f"\n📊 Batch evaluation completed:")
    print(f"   Total rows: {total_rows}")
    print(f"   Successful: {successful_evaluations}")
    print(f"   Failed (missing data): {failed_evaluations}")
    print(f"   JSON parsing errors: {json_errors}")
    print(f"   API errors: {api_errors}")
    print(f"   Timeout errors: {timeout_errors}")

    return df


if __name__ == "__main__":
    # === Usage examples ===
    # Basic usage

    # 1. Load the CSV file and prompt template for evaluation
    csv_path = r"C:\Users\Yijia Liu\Scribing Project\batch_processing_test\results_after100_20250724_192851.csv"
    prompt_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\prompt_templates\soap_note_eval_prompt.txt"

    # Check if files exist before trying to load them
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please check the file path and make sure the file exists.")
        exit(1)

    if not os.path.exists(prompt_path):
        print(f"❌ Prompt template file not found: {prompt_path}")
        print("Please check the file path and make sure the file exists.")
        exit(1)

    print(f"📂 Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from CSV file")

    print(f"📂 Loading prompt template: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    print("✅ Prompt template loaded successfully")

    # Debug: Show template info
    print(f"📊 Template length: {len(prompt_template)} characters")
    print(f"📝 Template preview (first 300 chars):")
    print(f"   {prompt_template[:300]}...")

    # Test the prompt rendering with sample data
    print("\n🧪 Testing prompt rendering...")
    try:
        sample_transcript = "Sample transcript for testing"
        sample_note = "Sample SOAP note for testing"
        test_prompt = render_prompt(prompt_template, sample_transcript, sample_note)
        print("✅ Prompt rendering test successful")
    except Exception as e:
        print(f"❌ Prompt rendering test failed: {e}")
        print("Please fix the prompt template before proceeding.")
        exit(1)

    # Define the profile ARN - Using Sonnet 4 for faster processing
    profile_arn = "arn:aws:bedrock:us-west-2:390402542319:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"

    print(f"🤖 Using model: Claude Sonnet 4 (faster processing)")

    # 2. Call Claude Sonnet 4 to conduct batch evaluation with checkpoint
    print("🚀 Starting batch evaluation with checkpoint functionality...")

    # Use checkpoint version for better reliability
    result_df = batch_evaluate_claude_with_checkpoint(
        df, prompt_template, profile_arn, 'evaluation_checkpoint_sonnet4.pkl'
    )

    # 3. Save the evaluated results to a new file
    final_output = "evaluated_results_with_claude_sonnet4_final.csv"
    result_df.to_csv(final_output, index=False)
    print(f"💾 Final results saved to: {final_output}")

    # Check failed cases
    failed_cases = result_df[result_df['evaluation_claude_sonnet4'].apply(
        lambda x: x.get('processing_status') != 'completed'
    )]

    # Check JSON load failure cases
    json_errors = result_df[result_df['evaluation_claude_sonnet4'].apply(
        lambda x: x.get('processing_status') == 'json_error'
    )]

    print(f"\n📋 Analysis Summary:")
    print(f"   Failed cases: {len(failed_cases)}")
    print(f"   JSON errors: {len(json_errors)}")

    # Basic usage without checkpoint (if you prefer)
    # result_df = batch_evaluate_claude(df, prompt_template, profile_arn)