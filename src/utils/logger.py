"""
Logging module for No-Harm-VLLM - Parse observability and debugging.
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ParseLogEntry:
    """Log entry for a single parse attempt."""
    timestamp: str
    instance_id: str
    juror: str
    dimension: Optional[str] = None
    raw_output: str = ""
    raw_output_length: int = 0
    strategies_tried: List[str] = field(default_factory=list)
    final_strategy: Optional[str] = None
    success: bool = False
    scores: Optional[Dict[str, float]] = None
    failure_reason: Optional[str] = None
    retries: int = 0
    contains_think_tags: bool = False
    contains_json: bool = False


class ParseLogger:
    """
    Logs every parse attempt for observability and debugging.

    Stores full raw outputs and extraction details for analysis.
    """

    def __init__(self, output_dir: str = "logs"):
        """Initialize logger with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.entries: List[ParseLogEntry] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up file and console logging."""
        log_file = self.output_dir / f"parse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_parse_attempt(
        self,
        instance_id: str,
        juror: str,
        dimension: Optional[str],
        raw_output: str,
        strategies_tried: List[str],
        final_strategy: Optional[str],
        success: bool,
        scores: Optional[Dict[str, float]] = None,
        failure_reason: Optional[str] = None,
        retries: int = 0,
    ) -> None:
        """
        Log a parse attempt.

        Args:
            instance_id: ID of the instance being scored
            juror: Name of the jury member
            dimension: Dimension being scored (or None for full response)
            raw_output: Raw model output
            strategies_tried: List of strategies attempted
            final_strategy: Final strategy used
            success: Whether parsing succeeded
            scores: Extracted scores if successful
            failure_reason: Reason for failure if unsuccessful
            retries: Number of retries attempted
        """
        entry = ParseLogEntry(
            timestamp=datetime.now().isoformat(),
            instance_id=instance_id,
            juror=juror,
            dimension=dimension,
            raw_output=raw_output[:1000],  # Truncate for storage
            raw_output_length=len(raw_output),
            strategies_tried=strategies_tried,
            final_strategy=final_strategy,
            success=success,
            scores=scores,
            failure_reason=failure_reason,
            retries=retries,
            contains_think_tags="<think>" in raw_output.lower(),
            contains_json="{" in raw_output,
        )

        self.entries.append(entry)

        # Log at appropriate level
        if success:
            self.logger.info(
                f"Parse successful: {juror} on {instance_id} "
                f"(strategies: {len(strategies_tried)}, retries: {retries})"
            )
        else:
            self.logger.warning(
                f"Parse failed: {juror} on {instance_id} "
                f"(final strategy: {final_strategy}, reason: {failure_reason})"
            )

    def log_batch(
        self,
        instance_id: str,
        juror: str,
        results: Dict[str, Dict],
    ) -> None:
        """
        Log batch parsing results for all dimensions.

        Args:
            instance_id: ID of the instance
            juror: Name of the jury member
            results: Dict mapping dimension to parse result
        """
        for dimension, result in results.items():
            self.log_parse_attempt(
                instance_id=instance_id,
                juror=juror,
                dimension=dimension,
                raw_output=result.get("raw_output", ""),
                strategies_tried=result.get("strategies_tried", []),
                final_strategy=result.get("final_strategy"),
                success=result.get("success", False),
                scores=result.get("scores"),
                failure_reason=result.get("failure_reason"),
                retries=result.get("retries", 0),
            )

    def save(self, filename: Optional[str] = None) -> Path:
        """Save logs to JSON file."""
        if filename is None:
            filename = f"parse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.output_dir / filename

        # Convert entries to dictionaries
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(self.entries),
            "success_count": sum(1 for e in self.entries if e.success),
            "failure_count": sum(1 for e in self.entries if not e.success),
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "instance_id": e.instance_id,
                    "juror": e.juror,
                    "dimension": e.dimension,
                    "raw_output_length": e.raw_output_length,
                    "strategies_tried": e.strategies_tried,
                    "final_strategy": e.final_strategy,
                    "success": e.success,
                    "scores": e.scores,
                    "failure_reason": e.failure_reason,
                    "retries": e.retries,
                    "contains_think_tags": e.contains_think_tags,
                    "contains_json": e.contains_json,
                }
                for e in self.entries
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Parse logs saved to {output_path}")
        return output_path

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.entries:
            return {}

        total = len(self.entries)
        success = sum(1 for e in self.entries if e.success)
        failure = total - success

        # Count think tags
        think_count = sum(1 for e in self.entries if e.contains_think_tags)

        # Count JSON in output
        json_count = sum(1 for e in self.entries if e.contains_json)

        # Average retries for failures
        failure_retries = [
            e.retries for e in self.entries if not e.success
        ]
        avg_retries = (
            sum(failure_retries) / len(failure_retries)
            if failure_retries else 0
        )

        return {
            "total_attempts": total,
            "success_count": success,
            "failure_count": failure,
            "success_rate": round(100 * success / total, 1),
            "think_tag_count": think_count,
            "json_count": json_count,
            "avg_retries_on_failure": round(avg_retries, 2),
        }
