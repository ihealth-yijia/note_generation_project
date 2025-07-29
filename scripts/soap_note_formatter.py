"""
SOAP Note Formatter Module

This module provides functionality to reformat existing SOAP notes using OpenAI's API.
It takes a SOAP note and reformats it according to specified detail level and organization style.
"""

import os
from typing import Optional
import openai
from dotenv import load_dotenv

load_dotenv()


class SOAPFormatter:
    """
    A class to format existing SOAP notes using OpenAI's API.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "chatgpt-4o-latest"):
        """
        Initialize the SOAP formatter.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
        model_name : str, default "chatgpt-4o-latest"
            The OpenAI model to use for formatting.
        """
        self.model_name = model_name

        if api_key:
            openai.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError(
                    "OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")

    def format_soap_note(self, prompt_template: str, current_note: str,
                         detail_level: str, organization_style: str,
                         temperature: float = 0.2, max_tokens: int = 5000) -> str:
        """
        Format a SOAP note using OpenAI API.

        Parameters
        ----------
        prompt_template : str
            The prompt template containing {current_note}, {detail_level}, and {organization_style} placeholders.
        current_note : str
            The existing SOAP note to be formatted.
        detail_level : str
            The desired detail level: "brief", "standard", "detailed", or "comprehensive".
        organization_style : str
            The desired organization style: "paragraph" or "bullet_point".
        temperature : float, default 0.2
            Temperature parameter for generation (0.0 to 1.0).
        max_tokens : int, default 5000
            Maximum tokens for the generated response.

        Returns
        -------
        str
            The formatted SOAP note from OpenAI.
        """
        # Validate parameters
        valid_detail_levels = ["brief", "standard", "detailed", "comprehensive"]
        valid_organization_styles = ["paragraph", "bullet_point"]

        if detail_level not in valid_detail_levels:
            raise ValueError(f"detail_level must be one of: {valid_detail_levels}")

        if organization_style not in valid_organization_styles:
            raise ValueError(f"organization_style must be one of: {valid_organization_styles}")

        system_message = "You are a medical documentation formatting expert and must follow all formatting and citation rules exactly."
        user_message = prompt_template.format(
            current_note=current_note,
            detail_level=detail_level,
            organization_style=organization_style
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        return response.choices[0].message.content


# Convenience function for standalone usage
def format_soap_note(prompt_template: str, current_note: str,
                     detail_level: str, organization_style: str,
                     api_key: Optional[str] = None, model_name: str = "chatgpt-4o-latest",
                     temperature: float = 0.2, max_tokens: int = 5000) -> str:
    """
    Convenience function to format a SOAP note.

    Parameters
    ----------
    prompt_template : str
        The prompt template containing {current_note}, {detail_level}, and {organization_style} placeholders.
    current_note : str
        The existing SOAP note to be formatted.
    detail_level : str
        The desired detail level: "brief", "standard", "detailed", or "comprehensive".
    organization_style : str
        The desired organization style: "paragraph" or "bullet_point".
    api_key : str, optional
        OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
    model_name : str, default "chatgpt-4o-latest"
        The OpenAI model to use for formatting.
    temperature : float, default 0.2
        Temperature parameter for generation (0.0 to 1.0).
    max_tokens : int, default 5000
        Maximum tokens for the generated response.

    Returns
    -------
    str
        The formatted SOAP note.
    """
    formatter = SOAPFormatter(api_key=api_key, model_name=model_name)
    return formatter.format_soap_note(prompt_template, current_note, detail_level,
                                      organization_style, temperature, max_tokens)

import pandas as pd
def test_real_data():
    """Test SOAPFormatter with real prompt template and CSV data"""

    # 1. Read the prompt template
    prompt_template_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\prompt_templates\soap_step2_generation_prompt.txt"

    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print("✓ Successfully loaded prompt template")
    except Exception as e:
        print(f"✗ Error loading prompt template: {e}")
        return

    # 2. Read the CSV file
    csv_path = r"C:\Users\Yijia Liu\Scribing Project\note_generation_project\eval_results\evaluated_results_claude_sonnet4_all.csv"

    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return

    # 3. Filter for transcript_id = 'A074'
    filtered_df = df[df['transcript_id'] == 'A074']
    current_note = filtered_df['normalized_soap_note'].iloc[0]
    print("✓ Successfully extracted normalized_soap_note")
    print(f"Note length: {len(current_note)} characters")

    # 5. Show preview of the note
    print("\n" + "=" * 50)
    print("PREVIEW OF CURRENT NOTE:")
    print("=" * 50)
    print(current_note[:500] + "..." if len(current_note) > 500 else current_note)

    # 6. Test formatting with different configurations
    test_configs = [
        ("standard", "bullet_point"),
        ("standard", "paragraph")
    ]

    for detail_level, organization_style in test_configs:
        print(f"\n{'=' * 60}")
        print(f"TESTING: {detail_level.upper()} + {organization_style.upper()}")
        print("=" * 60)

        try:
            result = format_soap_note(
                prompt_template=prompt_template,
                current_note=current_note,
                detail_level=detail_level,
                organization_style=organization_style,
                temperature=0.2,
                max_tokens=5000
            )

            print("✓ Successfully generated formatted note")
            print(f"Result length: {len(result)} characters")
            print("\nFORMATTED NOTE:")
            print("-" * 40)
            print(result)

            # Check if citations are preserved
            import re
            citations = re.findall(r'\[T\d+[^]]*\]\[SPEAKER \d+\]', result)
            print(f"\n📊 Citations found: {len(citations)}")
            if len(citations) > 0:
                print(f"Sample citations: {citations[:3]}")

        except Exception as e:
            print(f"✗ Error during formatting: {e}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    test_real_data()