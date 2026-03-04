"""
Model-specific profiles for preprocessing jury model outputs before parsing.

Each model has characteristic output patterns that need to be cleaned
before the universal ScoreExtractor can parse them.

Based on README_Qwen2.5.md - the actual 5 jury models are:
1. Ministral-14B - response generator + jury (28GB)
2. Gemma3-27B - Google Gemma 3 (54GB)
3. Nemotron-30B - NVIDIA reasoning (60GB)
4. OLMo-32B-Think - Open-source reasoning (64GB)
5. Qwen2.5-Coder-7B - Code-focused model (15GB)
"""

import re
from typing import Dict, List, Optional, Tuple


# Model-specific cleaning profiles
# Using the EXACT model names from the actual jury configuration
MODEL_PROFILES = {
    "ministral-14b": {
        "supports_json_mode": True,
        "has_thinking_mode": False,
        "preferred_format": "json",
        "system_suffix": "",
        "strip_patterns": [],
        "trim_prefix": True,
        "trim_prefix_patterns": [
            r'^[^\{]*',  # Strip everything before first {
        ],
    },
    "gemma3-27b": {
        "supports_json_mode": True,
        "has_thinking_mode": False,
        "preferred_format": "json",
        "system_suffix": "",
        "strip_patterns": [],
        "trim_prefix": True,
        "trim_prefix_patterns": [r'^[^\{]*'],
    },
    "nemotron-30b": {
        "supports_json_mode": True,
        "has_thinking_mode": False,
        "preferred_format": "json",
        "system_suffix": "\nRespond with ONLY valid JSON. No explanation.",
        "strip_patterns": [
            r'<\|.*?\|>',  # Remove NVIDIA special tokens
            r'<\|begin_of_text\|>',
            r'<\|end_of_text\|>',
            r'<\|reserved_special_token_.*?\|>',
        ],
        "trim_prefix": True,
        "trim_prefix_patterns": [
            r'^[^\{]*',  # Strip everything before first {
            r'^[Ss]ure[.,!:\s]*',  # Remove "Sure," type prefixes
            r'^[Oo]kay[.,!:\s]*',  # Remove "Okay," type prefixes
        ],
    },
    "olmo-32b": {
        "supports_json_mode": False,
        "has_thinking_mode": True,
        "preferred_format": "json_after_think",
        "system_suffix": "\nAfter your reasoning, output ONLY a JSON object.",
        "strip_patterns": [
            r'<think>.*?</think>',  # Remove think blocks
            r'<think>.*?</think>',
        ],
        "trim_prefix": True,
        "trim_prefix_patterns": [
            r'^[^\{]*',  # Strip everything before first {
        ],
    },
    "qwen2.5-coder-7b": {
        "supports_json_mode": True,
        "has_thinking_mode": True,
        "preferred_format": "json",
        "system_suffix": "\nRespond with ONLY valid JSON. No explanation.",
        "strip_patterns": [
            r'<think>.*?</think>',  # Remove think blocks
            r'<think>.*?</think>',
        ],
        "trim_prefix": True,
        "trim_prefix_patterns": [
            r'^[^\{]*',  # Strip everything before first {
        ],
    },
}


def clean_output(raw: str, model_name: str) -> str:
    """
    Strip model-specific artifacts before universal parsing.

    Args:
        raw: Raw output from the model
        model_name: Name of the model that generated the output

    Returns:
        Cleaned output ready for parsing
    """
    profile = MODEL_PROFILES.get(model_name, MODEL_PROFILES.get("ministral-14b", {}))

    cleaned = raw

    # Apply model-specific strip patterns
    for pattern in profile.get("strip_patterns", []):
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

    # Strip prefix if configured
    if profile.get("trim_prefix", False):
        for pattern in profile.get("trim_prefix_patterns", []):
            cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.DOTALL)

    # Universal cleaning
    cleaned = cleaned.strip()

    return cleaned


def get_model_profile(model_name: str) -> Dict:
    """
    Get the profile for a model, with fallback to ministral-14b.

    Args:
        model_name: Name of the model

    Returns:
        Model profile dictionary
    """
    return MODEL_PROFILES.get(model_name, MODEL_PROFILES.get("ministral-14b", {}))


def has_thinking_mode(model_name: str) -> bool:
    """Check if model has thinking mode that needs tag stripping."""
    return MODEL_PROFILES.get(model_name, {}).get("has_thinking_mode", False)


def has_json_mode_support(model_name: str) -> bool:
    """Check if model supports JSON mode in inference."""
    return MODEL_PROFILES.get(model_name, {}).get("supports_json_mode", False)


def extract_thought(raw_output: str, model_name: str) -> Tuple[str, str]:
    """
    Extract the thought/reasoning part from model output.

    Args:
        raw_output: Full raw output
        model_name: Name of the model

    Returns:
        Tuple of (thought_content, response_content)
    """
    if not has_thinking_mode(model_name):
        return ("", raw_output)

    profile = MODEL_PROFILES.get(model_name, {})

    # Try different think block patterns
    patterns = [
        r'<think>(.*?)</think>',
        r'<think>(.*?)</think>',
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            response = raw_output.replace(match.group(0), "").strip()
            return (thought, response)

    return ("", raw_output)
