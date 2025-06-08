# wake_word/kokoro_tts.py

from pathlib import Path
import torch
import torchaudio
from typing import Union

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from wake_word.tts_base import TTSEngine


class KokoroTTSEngine(TTSEngine):
    def __init__(
        self,
        model_id: str = "k2-fsa/kokoro-82m-en",
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 16000,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.sample_rate = sample_rate

        # Load model & processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def synthesize(self, text: str, output_path: Union[str, Path] = None) -> torch.Tensor:
        """
        Synthesizes speech from text.
        Returns: waveform (Tensor) or saves to file if output_path is given.
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio = self.model.generate(**inputs)
        waveform = audio.cpu().squeeze(0)  # (1, N) -> (N,)

        if output_path:
            torchaudio.save(str(output_path), waveform.unsqueeze(0), self.sample_rate)

        return waveform

    def save_to_file(self, text: str, filepath: Union[str, Path]) -> None:
        self.synthesize(text, output_path=filepath)
