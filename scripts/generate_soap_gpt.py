"""
SOAP Note Generator Module

This module provides functionality to generate and normalize SOAP notes using OpenAI's API.
It takes a prompt template and indexed transcript as input and returns a normalized SOAP note.
"""

import os
import re
from typing import List, Optional
import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv()

class SOAPGenerator:
    """
    A class to generate and normalize SOAP notes using OpenAI's API.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "chatgpt-4o-latest"):
        """
        Initialize the SOAP generator.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
        model_name : str, default "chatgpt-4o-latest"
            The OpenAI model to use for generation.
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

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text for the current model.

        Parameters
        ----------
        text : str
            Text to count tokens for.

        Returns
        -------
        int
            Number of tokens in the text.
        """
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # fallback for unknown models
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def generate_soap_note(self, prompt_template: str, indexed_transcript: str,
                           temperature: float = 0.2, max_tokens: int = 5000) -> str:
        """
        Generate a SOAP note using OpenAI API.

        Parameters
        ----------
        prompt_template : str
            The prompt template containing a {transcript} placeholder.
        indexed_transcript : str
            The indexed transcript with [T##] and [SPEAKER #] tags.
        temperature : float, default 0.2
            Temperature parameter for generation (0.0 to 1.0).
        max_tokens : int, default 1500
            Maximum tokens for the generated response.

        Returns
        -------
        str
            The generated SOAP note from OpenAI.
        """
        system_message = "You are a meticulous medical scribe and must follow all tagging rules exactly."
        user_message = prompt_template.format(transcript=indexed_transcript)

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


    def normalize_soap_note(self, raw_note: str) -> str:
        """
        Convert GPT markdown/bullet note to flat format.

        Parameters
        ----------
        raw_note : str
            The raw GPT output string with markdown formatting.

        Returns
        -------
        str
            Normalized note in flat format.
        """
        # Regex helpers
        heading_re = re.compile(
            r"^\s*\**\s*(Subjective|Objective|Assessment|Plan|Speaker Role Mapping)\s*[:：]?\s*\**\s*$",
            re.I,
        )
        bullet_re = re.compile(r"^\s*(?:[-‑–—•]\s+|\d+\.\s+)")

        def clean_line(line: str) -> str:
            """Remove bullet, trim, collapse spaces, normalize dashes in T-ranges."""
            line = bullet_re.sub("", line)  # strip bullets / numbers
            line = re.sub(r"[ \t]+", " ", line)  # collapse weird spaces
            # Replace en/em dashes ONLY inside [Tx–Ty] ranges
            line = re.sub(r"(\[T\d+)–(\d+\])", r"\1-\2", line)
            line = re.sub(r"(\[T\d+)—(\d+\])", r"\1-\2", line)
            return line.strip()

        def flush_buffer(buf: List[str], out: List[str]):
            """Move accumulated sentence(s) to output if buffer not empty."""
            if not buf:
                return
            joined = " ".join(buf).strip()
            if joined:
                out.append(joined)
            buf.clear()

        lines = raw_note.splitlines()
        cleaned_lines: List[str] = []
        buffer: List[str] = []  # accumulates wrapped sentence fragments
        current_section = None

        for ln in lines:
            # Check if this is a heading
            m = heading_re.match(ln)
            if m:
                # flush any buffered sentence from previous section
                flush_buffer(buffer, cleaned_lines)

                current_section = m.group(1).title()  # Proper-case
                cleaned_lines.append(f"{current_section}:")
                continue

            # ignore blank lines
            if not ln.strip():
                # allow wrapped paragraphs to break here
                flush_buffer(buffer, cleaned_lines)
                continue

            # Normal content line
            fragment = clean_line(ln)

            # If GPT wrapped a long bullet across multiple physical lines,
            # keep collecting until we see the next bullet or blank line.
            bullet_start = bullet_re.match(ln) is not None
            if bullet_start and buffer:
                flush_buffer(buffer, cleaned_lines)

            buffer.append(fragment)

        # flush remainder
        flush_buffer(buffer, cleaned_lines)

        return "\n".join(cleaned_lines)

    def generate_and_normalize(self, prompt_template: str, indexed_transcript: str,
                               temperature: float = 0.2, max_tokens: int = 1500) -> str:
        """
        Generate and normalize a SOAP note in one step.

        Parameters
        ----------
        prompt_template : str
            The prompt template containing a {transcript} placeholder.
        indexed_transcript : str
            The indexed transcript with [T##] and [SPEAKER #] tags.
        temperature : float, default 0.2
            Temperature parameter for generation (0.0 to 1.0).
        max_tokens : int, default 1500
            Maximum tokens for the generated response.

        Returns
        -------
        str
            The normalized SOAP note.
        """
        raw_note = self.generate_soap_note(prompt_template, indexed_transcript,
                                           temperature, max_tokens)
        return self.normalize_soap_note(raw_note)


# Convenience functions for standalone usage
def generate_soap_note(prompt_template: str, indexed_transcript: str,
                       api_key: Optional[str] = None, model_name: str = "chatgpt-4o-latest",
                       temperature: float = 0.2, max_tokens: int = 1500) -> str:
    """
    Convenience function to generate and normalize a SOAP note.

    Parameters
    ----------
    prompt_template : str
        The prompt template containing a {transcript} placeholder.
    indexed_transcript : str
        The indexed transcript with [T##] and [SPEAKER #] tags.
    api_key : str, optional
        OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
    model_name : str, default "chatgpt-4o-latest"
        The OpenAI model to use for generation.
    temperature : float, default 0.2
        Temperature parameter for generation (0.0 to 1.0).
    max_tokens : int, default 1500
        Maximum tokens for the generated response.

    Returns
    -------
    str
        The normalized SOAP note.
    """
    generator = SOAPGenerator(api_key=api_key, model_name=model_name)
    return generator.generate_and_normalize(prompt_template, indexed_transcript,
                                            temperature, max_tokens)


if __name__ == "__main__":
    # Example usage
    sample_prompt = """
    Generate a SOAP note from the following transcript:
    {transcript}
    """

    sample_transcript = """
    [T1][SPEAKER 0]: How are you feeling today?
    [T2][SPEAKER 1]: I'm feeling good, doctor.
    """

    try:
        result = generate_soap_note(sample_prompt, sample_transcript)
        print("Generated SOAP Note:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable.")