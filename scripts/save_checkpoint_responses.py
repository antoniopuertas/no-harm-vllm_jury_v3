#!/usr/bin/env python3
"""
Script to manually save checkpoint responses if needed.
This would need to be integrated into the main evaluation script.
"""

import json
import sys
from pathlib import Path

def save_responses_checkpoint(responses, checkpoint_file):
    """Save responses to a checkpoint file"""
    checkpoint_file = Path(checkpoint_file)

    # Load existing checkpoint
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {}

    # Add responses
    checkpoint['responses'] = responses
    checkpoint['num_responses'] = len(responses)

    # Save with _responses suffix
    response_checkpoint = checkpoint_file.parent / f"{checkpoint_file.stem}_responses.json"
    with open(response_checkpoint, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"Saved {len(responses)} responses to: {response_checkpoint}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python save_checkpoint_responses.py <checkpoint_file> <num_responses>")
        print("Example: python save_checkpoint_responses.py .checkpoint_medmcqa.json 1300")
        sys.exit(1)

    checkpoint_file = sys.argv[1]
    num_responses = int(sys.argv[2])

    print(f"This script needs to be run from within the evaluation process")
    print(f"or responses need to be extracted from memory")
