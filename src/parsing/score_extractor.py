"""
Score Extractor - Engine-agnostic extraction of dimension scores from jury model outputs.

This module provides a 4-strategy cascade parser that attempts multiple approaches
to extract scores from jury model responses, falling back to explicit None on total failure
instead of silently defaulting to 0.5.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ExtractionStrategy(Enum):
    """Strategies used for score extraction."""
    DIRECT_JSON = "direct_json"
    FENCED_JSON = "fenced_json"
    REGEX_PAIRS = "regex_pairs"
    LINE_BY_LINE = "line_by_line"
    NONE = "none"


@dataclass
class ExtractionResult:
    """Result of score extraction attempt."""
    scores: Optional[Dict[str, float]]
    strategy: ExtractionStrategy
    success: bool
    raw_output: str
    retries: int = 0
    failure_reason: Optional[str] = None


class ScoreExtractor:
    """
    Engine-agnostic score extraction with cascading fallback strategies.

    Attempts multiple parsing strategies in order:
    1. Direct JSON parsing (entire output is JSON)
    2. Fenced JSON extraction (extract from ```json ... ``` blocks)
    3. Regex-based dimension: score pair matching
    4. Line-by-line extraction (find 7 floats in [0,1] range)

    Returns None on total failure - NEVER returns 0.5 as a default.
    """

    DIMENSIONS = [
        "informational", "social", "psychological",
        "autonomy", "economic", "privacy", "epistemic"
    ]

    # Dimension names with common variants for matching
    DIMENSION_ALIASES = {
        "informational": ["informational", "information", "info"],
        "psychological": ["psychological", "psych", "mental"],
        "social": ["social", "societal"],
        "autonomy": ["autonomy", "paternalism", "choice"],
        "economic": ["economic", "economy", "cost", "financial"],
        "privacy": ["privacy", "confidentiality", "hipaa"],
        "epistemic": ["epistemic", "epistemology", "expertise"],
    }

    def __init__(self, max_retries: int = 2):
        """
        Initialize score extractor.

        Args:
            max_retries: Maximum number of retry attempts with reformulated prompts
        """
        self.max_retries = max_retries
        self.extraction_history: List[ExtractionResult] = []

    def extract(self, raw_output: str) -> ExtractionResult:
        """
        Try multiple extraction strategies. Return ExtractionResult.

        Args:
            raw_output: Raw text response from jury member

        Returns:
            ExtractionResult with scores, strategy used, and success status
            On total failure: scores=None, success=False
        """
        raw_output = raw_output.strip()

        # Try each strategy in cascade
        for strategy in [
            self._try_direct_json,
            self._try_fenced_json,
            self._try_regex_pairs,
            self._try_line_by_line,
        ]:
            result = strategy(raw_output)
            if result is not None and self._validate(result):
                return ExtractionResult(
                    scores=result,
                    strategy=strategy.__name__.replace("_try_", "").upper(),
                    success=True,
                    raw_output=raw_output,
                    retries=0
                )

        # All strategies failed
        return ExtractionResult(
            scores=None,
            strategy=ExtractionStrategy.NONE,
            success=False,
            raw_output=raw_output,
            retries=0,
            failure_reason="All extraction strategies exhausted"
        )

    def _try_direct_json(self, text: str) -> Optional[Dict[str, float]]:
        """Attempt to parse the entire output as JSON."""
        # Clean common whitespace issues
        text = text.strip()

        # Try to parse as JSON
        try:
            data = json.loads(text)
            return self._normalize(data)
        except (json.JSONDecodeError, TypeError):
            return None

    def _try_fenced_json(self, text: str) -> Optional[Dict[str, float]]:
        """Extract JSON from ```json ... ``` or ``` ... ``` blocks."""
        # Try to find JSON blocks
        patterns = [
            r'```json\s*\n?(.*?)\n?\s*```',
            r'```\s*\n?(.*?)\n?\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    candidate = match.group(1)
                    data = json.loads(candidate.strip())
                    return self._normalize(data)
                except (json.JSONDecodeError, IndexError):
                    continue
        return None

    def _try_regex_pairs(self, text: str) -> Optional[Dict[str, float]]:
        """Extract dimension: score pairs from free-form text."""
        scores = {}

        for dim_key, aliases in self.DIMENSION_ALIASES.items():
            for alias in aliases:
                # Match patterns like:
                # "informational: 0.3"
                # "informational = 0.3"
                # "informational harm: 0.3"
                # "informational:0.3" (no space)
                pattern = rf'{alias}[\s\w_-]*[\s:=]+\s*(0(?:\.\d+)?|1(?:\.0+)?)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        scores[dim_key] = float(match.group(1))
                        break  # Found this dimension, move to next
                    except ValueError:
                        continue

        return scores if len(scores) == len(self.DIMENSIONS) else None

    def _try_line_by_line(self, text: str) -> Optional[Dict[str, float]]:
        """Last resort: find any 7 floats in [0,1] range, map to dimensions."""
        # Find all numbers that look like floats in [0,1]
        # Use negative lookahead to avoid matching '1' in '1.5', '1.2', etc.
        floats = re.findall(r'\b(0\.\d+|0|1(?![\d\.]))\b', text)

        # Filter to valid range
        candidates = []
        for f in floats:
            try:
                val = float(f)
                if 0.0 <= val <= 1.0:
                    candidates.append(val)
            except ValueError:
                continue

        # Only use if exactly 7 values found
        if len(candidates) == 7:
            return dict(zip(self.DIMENSIONS, candidates))

        return None

    def _validate(self, scores: Dict[str, float]) -> bool:
        """Ensure all dimensions present and values in valid range."""
        score_keys = set(scores.keys())
        expected_keys = set(self.DIMENSIONS)

        # Check all dimensions present
        if score_keys != expected_keys:
            return False

        # Check all values in valid range
        for val in scores.values():
            if not isinstance(val, (int, float)):
                return False
            if not (0.0 <= val <= 1.0):
                return False

        return True

    def _normalize(self, data: Dict) -> Optional[Dict[str, float]]:
        """Normalize keys to lowercase, handle common variants."""
        normalized = {}

        for dim in self.DIMENSIONS:
            aliases = self.DIMENSION_ALIASES[dim]

            # Try to find matching key in data
            for key, val in data.items():
                key_lower = key.lower().replace("_", "").replace("-", "")

                # Check if any alias matches
                for alias in aliases:
                    if alias in key_lower:
                        try:
                            float_val = float(val)
                            # Validate range - out of range values cause failure
                            if not (0.0 <= float_val <= 1.0):
                                return None
                            normalized[dim] = float_val
                            break
                        except (ValueError, TypeError):
                            continue
                else:
                    continue
                break

        return normalized if len(normalized) == len(self.DIMENSIONS) else None

    def extract_with_retries(
        self,
        raw_outputs: List[str],
        reformulations: List[str] = None
    ) -> List[ExtractionResult]:
        """
        Extract scores from multiple outputs with retry logic.

        Args:
            raw_outputs: List of raw responses to parse
            reformulations: Optional list of reformulated prompts for retry

        Returns:
            List of ExtractionResults in same order as inputs
        """
        results = []

        for i, raw_output in enumerate(raw_outputs):
            result = self.extract(raw_output)

            if not result.success and reformulations and i < len(reformulations):
                # Try with reformulated prompt
                for retry_num in range(1, self.max_retries + 1):
                    # This would be called with the re-generated output
                    # For now, we just record that retry would be needed
                    pass

            self.extraction_history.append(result)
            results.append(result)

        return results
