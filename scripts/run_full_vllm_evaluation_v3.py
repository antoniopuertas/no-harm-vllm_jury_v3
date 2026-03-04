#!/usr/bin/env python3
"""
Improved Full Dataset Evaluation Script with Response Checkpointing

This version saves responses to checkpoint during generation,
so you can stop and resume without losing progress.

Key improvements:
- Saves responses incrementally to checkpoint file
- Can resume from saved responses
- Doesn't regenerate responses on resume

Usage: Same as run_full_vllm_evaluation.py
"""

import sys
import os
import time
import argparse
import logging
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.vllm_engine import VLLMEngine
from src.inference.model_manager import ModelManager
from src.evaluation.multi_dim_jury import MultiDimensionalJuryScorer
from src.data.dataset_loaders import get_dataset_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/full_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Global state
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.warning(f"\nShutdown signal received (signal {signum})")
    logger.info("Will complete current batch and save checkpoint...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

DATASET_SIZES = {
    "pubmedqa": 1000,
    "medqa": 1273,
    "medmcqa": 4183
}


def save_checkpoint(
    checkpoint_file: Path,
    dataset_name: str,
    current_idx: int,
    total_samples: int,
    responses: List[str] = None,
    jury_results: List[List[Dict]] = None,
    phase: str = "response_generation"
):
    """
    Save evaluation checkpoint with responses

    Args:
        checkpoint_file: Path to checkpoint file
        dataset_name: Dataset name
        current_idx: Current sample index
        total_samples: Total samples
        responses: Generated responses (saved during phase 1)
        jury_results: Jury scoring results (saved during phase 2)
        phase: Current phase ("response_generation" or "jury_scoring")
    """
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "phase": phase,
        "current_sample": current_idx,
        "total_samples": total_samples,
        "status": "in_progress"
    }

    if responses is not None:
        checkpoint["responses"] = responses
        checkpoint["num_responses"] = len(responses)

    if jury_results is not None:
        checkpoint["jury_results"] = jury_results

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: {checkpoint_file} (phase: {phase}, progress: {current_idx}/{total_samples})")


def load_checkpoint(checkpoint_file: Path) -> Optional[Dict]:
    """Load checkpoint if exists"""
    if not checkpoint_file.exists():
        return None

    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint from phase: {checkpoint.get('phase', 'unknown')}")
        if 'responses' in checkpoint:
            logger.info(f"  - Found {checkpoint.get('num_responses', 0)} saved responses")
        return checkpoint


def generate_responses_with_checkpoint(
    engine: VLLMEngine,
    model_name: str,
    instances: List[dict],
    checkpoint_file: Path,
    dataset_name: str,
    batch_size: int = 32,
    checkpoint_interval: int = 100
) -> List[str]:
    """
    Generate responses with incremental checkpointing
    """
    logger.info(f"Generating responses for {len(instances)} instances...")

    # Check for existing checkpoint with responses
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint and 'responses' in checkpoint:
        saved_responses = checkpoint['responses']
        start_idx = len(saved_responses)
        logger.info(f"Resuming from {start_idx} saved responses")
        all_responses = saved_responses.copy()
    else:
        start_idx = 0
        all_responses = []

    # Extract questions
    questions = []
    for instance in instances:
        question = instance.get("question", "")
        if not question:
            logger.warning(f"Instance {instance.get('id', 'unknown')} has no question")
            questions.append("")
        else:
            questions.append(f"Answer this medical question concisely: {question}")

    # Generate remaining responses
    start_time = time.time()

    for i in range(start_idx, len(questions), batch_size):
        if shutdown_requested:
            logger.info("Shutdown requested during response generation")
            save_checkpoint(
                checkpoint_file, dataset_name, len(all_responses), len(instances),
                responses=all_responses, phase="response_generation"
            )
            break

        batch = questions[i:i+batch_size]
        valid_batch = [q for q in batch if q]

        if valid_batch:
            responses = engine.generate_batch(
                model_name,
                valid_batch,
                temperature=0.0,
                max_tokens=1024
            )

            response_idx = 0
            for q in batch:
                if q:
                    all_responses.append(responses[response_idx])
                    response_idx += 1
                else:
                    all_responses.append("")

        current_idx = len(all_responses)

        # Save checkpoint at intervals
        if current_idx % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_file, dataset_name, current_idx, len(instances),
                responses=all_responses, phase="response_generation"
            )

        if current_idx % 100 == 0 or current_idx == len(questions):
            elapsed = time.time() - start_time
            logger.info(f"  Generated {current_idx}/{len(questions)} responses "
                       f"({elapsed:.1f}s, {elapsed/(current_idx-start_idx):.3f}s/instance)")

    # Final checkpoint
    save_checkpoint(
        checkpoint_file, dataset_name, len(all_responses), len(instances),
        responses=all_responses, phase="response_generation_complete"
    )

    return all_responses


# Import other functions from original script
# (score_with_jury_batch, aggregate_scores, save_results would be imported here)
# For brevity, I'm showing the key improvement: response checkpointing

if __name__ == "__main__":
    print("This is an improved version template.")
    print("To use it, you would need to copy the full script and integrate")
    print("the checkpoint improvements shown above.")
