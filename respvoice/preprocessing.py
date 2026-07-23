"""
Audio preprocessing: wav → log-Mel spectrogram.
Uses librosa for audio I/O (avoids torchaudio Windows DLL issues).
Unifies respiratory sounds and voice into the same 64×T representation.
"""

import numpy as np
import torch
import librosa


class AudioPreprocessor:
    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 64,
        win_ms: float = 64.0,
        hop_ms: float = 32.0,
        target_sec: float = 8.0,
        f_min: float = 50.0,
        f_max: float = 8000.0,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.target_len = int(target_sec * sr)
        self.win_length = int(win_ms * sr / 1000)
        self.hop_length = int(hop_ms * sr / 1000)
        self.f_min = f_min
        self.f_max = f_max

    def load(self, path: str) -> np.ndarray:
        """Load audio file → mono float32 at target sr."""
        wav, _ = librosa.load(path, sr=self.sr, mono=True)
        return wav  # (T,) float32

    def to_mel(self, wav: np.ndarray) -> torch.Tensor:
        """wav (T,) numpy → log-Mel (1, n_mels, T') Tensor."""
        wav = self._pad_or_crop(wav)
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )
        mel = np.log(mel + 1e-6).astype(np.float32)
        mel = self._normalize(mel)
        return torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T')

    def __call__(self, path: str) -> torch.Tensor:
        return self.to_mel(self.load(path))

    def from_tensor(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """Accepts (T,), (1,T), or (C,T) float Tensors."""
        wav = wav_tensor.numpy()
        if wav.ndim > 1:
            wav = wav.mean(0)
        return self.to_mel(wav.astype(np.float32))

    def _pad_or_crop(self, wav: np.ndarray) -> np.ndarray:
        L = len(wav)
        if L >= self.target_len:
            return wav[: self.target_len]
        # repeat-pad
        repeats = (self.target_len // L) + 1
        wav = np.tile(wav, repeats)
        return wav[: self.target_len]

    def _normalize(self, mel: np.ndarray) -> np.ndarray:
        mean = mel.mean()
        std = mel.std() + 1e-6
        return (mel - mean) / std
