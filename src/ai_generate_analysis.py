import json
from openai import OpenAI

from src.prompts import DATASET_ANALYSIS_PROMPT

def generate_dataset_analysis(
    profile: dict,
    sample_rows: list,
    api_key: str
) -> str:
    """
    Generate dataset insights using the official OpenAI Chat Completions API.
    """
    # 1. Initialize the official OpenAI client using the runtime session key
    client = OpenAI(api_key=api_key)

    # Serialize the profile metadata and row samples into clean JSON strings 
    # to guarantee structured, predictable readability for the LLM
    profile_str = json.dumps(profile, indent=2, default=str)
    sample_str = json.dumps(sample_rows, indent=2, default=str)

    # 2. Execute the Chat Completions request targeting the optimal cost-to-performance model
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # The premier model for lightweight structured data analytics
        messages=[
            {
                "role": "system",
                "content": DATASET_ANALYSIS_PROMPT
            },
            {
                "role": "user",
                "content": f"Dataset Profile Metadata:\n\n{profile_str}\n\nSample Rows:\n\n{sample_str}"
            }
        ],
        temperature=0.2  # Restrict creativity to ensure strict, deterministic statistical reporting
    )

    # 3. Extract and return the raw generated markdown content from the completion payload
    return response.choices[0].message.content

