from pathlib import Path
from abc import ABC, abstractmethod
from typing import List

from wake_word.config import ConfigManager


class TTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines"""

    @abstractmethod
    def generate_samples(
        self,
        text_variants: List[str],
        output_dir: Path,
        config: ConfigManager,
    ) -> int:
        """Generate audio samples for given text variants"""
        pass
