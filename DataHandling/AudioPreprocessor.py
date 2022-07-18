"""
The code is from IMS-Toucan toolkit.
Author: Florian Lux
Github: https://github.com/DigitalPhonetics/IMS-Toucan
"""
import librosa
import librosa.core as lb
import numpy
import numpy as np
import pyloudnorm as pyln
import torch
from torchaudio.transforms import MuLawDecoding
from torchaudio.transforms import MuLawEncoding
from torchaudio.transforms import Resample
from torchaudio.transforms import Vad as VoiceActivityDetection


class AudioPreprocessor:
    def __init__(
            self,
            input_sr,
            output_sr=None,
            melspec_buckets=80,
            hop_length=256,
            n_fft=1024,
            cut_silence=False,
    ):
        """
        The parameters are by default set up to do well
        on a 16kHz signal. A different sampling rate may
        require different hop_length and n_fft (e.g.
        doubling frequency --> doubling hop_length and
        doubling n_fft)
        """
        self.cut_silence = cut_silence
        self.sr = input_sr
        self.new_sr = output_sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mel_buckets = melspec_buckets
        self.vad = VoiceActivityDetection(sample_rate=input_sr)
        # ^^^^ This needs heavy tweaking if used, depends very much on the data
        self.mu_encode = MuLawEncoding()
        self.mu_decode = MuLawDecoding()
        self.meter = pyln.Meter(input_sr)
        self.final_sr = input_sr
        if output_sr is not None and output_sr != input_sr:
            self.resample = Resample(orig_freq=input_sr, new_freq=output_sr)
            self.final_sr = output_sr
        else:
            self.resample = lambda x: x

    def apply_mu_law(self, audio):
        """
        brings the audio down from 16 bit
        resolution to 8 bit resolution to
        make using softmax to predict a
        wave from it more feasible.
        !CAREFUL! transforms the floats
        between -1 and 1 to integers
        between 0 and 255. So that is good
        to work with, but bad to save/listen
        to. Apply mu-law decoding before
        saving or listening to the audio.
        """
        return self.mu_encode(audio)

    def cut_silence_from_beginning_and_end(self, audio):
        """
        applies cepstral voice activity
        detection and noise reduction to
        cut silence from the beginning
        and end of a recording
        """
        silence = torch.zeros([20000])
        no_silence_front = self.vad(
            torch.cat((silence, torch.Tensor(audio), silence), 0)
        )
        reversed_audio = torch.flip(no_silence_front, (0,))
        no_silence_back = self.vad(torch.Tensor(reversed_audio))
        unreversed_audio = torch.flip(no_silence_back, (0,))
        return unreversed_audio

    def to_mono(self, x):
        """
        make sure we deal with a 1D array
        """
        if len(x.shape) == 2:
            return lb.to_mono(numpy.transpose(x))
        else:
            return x

    def normalize_loudness(self, audio):
        """
        normalize the amplitudes according to
        their decibels, so this should turn any
        signal with different magnitudes into
        the same magnitude by analysing loudness
        """
        loudness = self.meter.integrated_loudness(audio)
        loud_normed = pyln.normalize.loudness(audio, loudness, -30.0)
        peak = numpy.amax(numpy.abs(loud_normed))
        peak_normed = numpy.divide(loud_normed, peak)
        return peak_normed

    def logmelfilterbank(self, audio, sampling_rate, fmin=40, fmax=8000, eps=1e-10):
        """
        Compute log-Mel filterbank
        """
        audio = audio.numpy()
        # get amplitude spectrogram
        x_stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window="hann",
            pad_mode="reflect",
        )
        spc = np.abs(x_stft).T
        # get mel basis
        fmin = 0 if fmin is None else fmin
        fmax = sampling_rate / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(
            sampling_rate, self.n_fft, self.mel_buckets, fmin, fmax
        )
        # apply log and return
        return torch.Tensor(
            np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))
        ).transpose(0, 1)

    def normalize_audio(self, audio):
        """
        one function to apply them all in an
        order that makes sense.
        """
        audio = self.to_mono(audio)
        audio = self.normalize_loudness(audio)
        if self.cut_silence:
            audio = self.cut_silence_from_beginning_and_end(audio)
        else:
            audio = torch.Tensor(audio)
        audio = self.resample(audio)
        return audio

    def audio_to_wave_tensor(self, audio, normalize=True, mulaw=False):
        if mulaw:
            if normalize:
                return self.apply_mu_law(self.normalize_audio(audio))
            else:
                return self.apply_mu_law(torch.Tensor(audio))
        else:
            if normalize:
                return self.normalize_audio(audio)
            else:
                return torch.Tensor(audio)

    def audio_to_mel_spec_tensor(self, audio, normalize=True):
        if normalize:
            return self.logmelfilterbank(
                audio=self.normalize_audio(audio), sampling_rate=self.new_sr
            )
        else:
            if isinstance(audio, torch.Tensor):
                return self.logmelfilterbank(audio=audio, sampling_rate=self.sr)
            return self.logmelfilterbank(
                audio=torch.Tensor(audio), sampling_rate=self.sr
            )
