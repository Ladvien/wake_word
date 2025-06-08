from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from wake_word.train import ConfigManager


class TTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines"""

    @abstractmethod
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: "ConfigManager") -> int:
        """Generate audio samples for given text variants"""
        pass
