import pandas as pd
import boto3
import json
import time

#from generate_soap import indexed_transcript


# === 1. Load system prompt template ===
def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template text from a local file"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# === 2. Render prompt with transcript and note ===
def render_prompt(prompt_template: str, indexed_transcript: str, normalized_note: str) -> str:
    """Insert actual transcript and note into the prompt template"""
    return prompt_template.format(
        indexed_transcript=indexed_transcript,
        normalized_note=normalized_note
    )

# === 3. Call Claude Opus 4 via AWS Bedrock new ===
def call_claude_opus_converse(
    rendered_prompt: str,
    profile_arn: str,
    profile_name: str = 'ihealth-test',
    region_name: str = 'us-west-2'
) -> str:
    """
    Call Claude Opus 4 via AWS Bedrock using Converse API + Inference Profile
    """
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime', region_name=region_name)

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


# === 4. Batch evaluation loop ===
def batch_evaluate_claude(df: pd.DataFrame, prompt_template: str) -> pd.DataFrame:
    """Iterate through each row of the DataFrame and evaluate using Claude Opus 4"""
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
            print(f"📝 Sending prompt for row {i + 1}...")

            response_text = call_claude_opus(full_prompt)

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
    df['evaluation_claude_opus4'] = results

    # Print summary with detailed status breakdown
    successful_evaluations = sum(1 for r in results if r.get('processing_status') == 'completed')
    failed_evaluations = sum(1 for r in results if r.get('processing_status') == 'failed')
    json_errors = sum(1 for r in results if r.get('processing_status') == 'json_error')
    api_errors = sum(1 for r in results if r.get('processing_status') == 'api_error')

    print(f"\n📊 Batch evaluation completed:")
    print(f"   Total rows: {total_rows}")
    print(f"   Successful: {successful_evaluations}")
    print(f"   Failed (missing data): {failed_evaluations}")
    print(f"   JSON parsing errors: {json_errors}")
    print(f"   API errors: {api_errors}")

    return df


# === Helper function: Save intermediate results ===
def batch_evaluate_claude_with_checkpoint(df: pd.DataFrame, prompt_template: str,
                                          checkpoint_file: str = None) -> pd.DataFrame:
    """Batch evaluation with checkpoint functionality to resume from interruption"""
    results = []
    total_rows = len(df)

    # Load existing results if checkpoint file exists
    start_idx = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            checkpoint_df = pd.read_pickle(checkpoint_file)
            if len(checkpoint_df) > 0:
                start_idx = len(checkpoint_df)
                results = checkpoint_df['evaluation_claude_opus4'].tolist()
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
            response_text = call_claude_opus(full_prompt)

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
            temp_df['evaluation_claude_opus4'] = results
            temp_df.to_pickle(checkpoint_file)
            print(f"💾 Checkpoint saved at row {i + 1}")

        # Delay
        if i < total_rows - 1:
            time.sleep(2)

    # Add results to DataFrame
    df = df.copy()
    df['evaluation_claude_opus4'] = results

    # Save final checkpoint
    if checkpoint_file:
        df.to_pickle(checkpoint_file)
        print(f"💾 Final results saved to {checkpoint_file}")

    # Print summary with detailed status breakdown
    successful_evaluations = sum(1 for r in results if r.get('processing_status') == 'completed')
    failed_evaluations = sum(1 for r in results if r.get('processing_status') == 'failed')
    json_errors = sum(1 for r in results if r.get('processing_status') == 'json_error')
    api_errors = sum(1 for r in results if r.get('processing_status') == 'api_error')

    print(f"\n📊 Batch evaluation completed:")
    print(f"   Total rows: {total_rows}")
    print(f"   Successful: {successful_evaluations}")
    print(f"   Failed (missing data): {failed_evaluations}")
    print(f"   JSON parsing errors: {json_errors}")
    print(f"   API errors: {api_errors}")

    return df


# === Usage examples ===
# Basic usage

#1. Load the CSV file and prompt template for evaluation
df = pd.read_csv(r"C:\Users\Yijia Liu\Scribing Project\batch_processing_test\test_results_20250724_141548.csv")
prompt_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\prompt_templates\soap_note_eval_prompt.txt"
with open(prompt_path, 'r', encoding='utf-8') as file:
    prompt_template = file.read()

# 2. call calude opus4 to conduct batch evaluation
result_df = batch_evaluate_claude(df, prompt_template)

# 3. Save the evaluated results to a new file
result_df.to_csv("evaluated_results_with_claude_opus4.csv", index=False)

# check failed cases
failed_cases = result_df[result_df['evaluation_claude_opus4'].apply(
    lambda x: x.get('processing_status') != 'completed'
)]

# check jason load failure cases查看JSON
json_errors = result_df[result_df['evaluation_claude_opus4'].apply(
    lambda x: x.get('processing_status') == 'json_error'
)]

# Usage with checkpoint functionality
# result_df = batch_evaluate_claude_with_checkpoint(df, prompt_template, 'evaluation_checkpoint.pkl')


"""
#test case - single transcript evaluation
# 1. Load the evaluation prompt template, indexed transcript and soap note
prompt_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\prompt_templates\evaluation_prompt.txt"
prompt_template = load_prompt_template(prompt_path)

transcript_path = r"C:\Users\Yijia Liu\Scribing Project\indexedmedical transcript\A074_numbered.txt"
with open(transcript_path, 'r', encoding='utf-8') as file:
    indexed_transcript = file.read()

note_path = r"C:\Users\Yijia Liu\Scribing Project\soap_note\soap_note_A074.txt"
with open(transcript_path, 'r', encoding='utf-8') as file:
    normalized_note = file.read()

# 2. create prompt
rendered_prompt = render_prompt(prompt_template, indexed_transcript, normalized_note)

# 3. call claude copus 4 model
response = call_claude_opus_converse(
    rendered_prompt,
    profile_arn="arn:aws:bedrock:us-west-2:390402542319:inference-profile/us.anthropic.claude-opus-4-20250514-v1:0",
    profile_name="ihealth-test"
)
print(response)

# batch note evaluation
#1. Load the CSV file for evaluation
df = pd.read_csv(r"C:\Users\Yijia Liu\Scribing Project\batch_processing_test\test_results_20250724_141548.csv")

# 3. Run batch evaluation
df_with_eval = batch_evaluate_claude(df, prompt_template)

# 4. Save the evaluated results to a new file
df_with_eval.to_csv("evaluated_results_with_claude_opus4.csv", index=False)"""

