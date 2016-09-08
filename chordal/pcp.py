import collections
import numpy as np
from scipy.io import wavfile


class Chromagrammer:
    """
    Calculates chromagrams, a.k.a. Pitch Class Profile (PCP) vectors.

    Calculates chromagrams, 12 dimensional vectors indicating the relative
    intensity of musical semitones across a time-domain signal.  The
    chromagrams are calculated for a given sampling frequency and FFT length.
    """
    ref_freqs = {n: 2**(n/12) * 440.0 for n in range(12)}

    def __init__(self, sampling_freq, fft_length):
        self.f_s = sampling_freq
        self.N = fft_length

    def fft_bin_to_freq(self, k):
        return self.f_s * k / self.N

    def spectrum_bin_to_pcp_index(self, k, f_ref):
        return np.floor(12 * np.log2(self.fft_bin_to_freq(k) / f_ref) % 12)

    def single_chroma(self, data, pcp_index):
        """Calculate the chromagram for a single time-domain sample"""
        mapping = (abs(np.fft.fft(data))**2
                   for k in range(1, int(self.N/2 - 1))
                   if self.spectrum_bin_to_pcp_index(
                       k, Chromagrammer.ref_freqs[pcp_index]) == pcp_index)
        return sum(sum(mapping))

    def full_chroma(self, data_frames):
        """Calculate chromagrams for successive time-domain samples"""
        for frame in data_frames:
            pcp = np.array([self.single_chroma(frame, p)
                            for p in Chromagrammer.ref_freqs])
            yield pcp/pcp.max()


class AudioProcessor:
    """
    Performs audio pre-processing,

    Pre-processes audio from a wavfile, combining the two channels and
    dividing the audio into overlapping frames.
    """

    def __init__(self, file_name, window_len_s=0.025, overlap_s=0.01):
        self.f_s, self.data = self.read_wavfile(file_name)

        self.frame_size = int(self.f_s * window_len_s)
        self.overlap = int(self.f_s * overlap_s)
        self.num_frames = int(len(self.data)/self.overlap)

    @staticmethod
    def read_wavfile(filename):
        f_s, raw_data = wavfile.read(filename)
        return f_s, raw_data[:, 0] + raw_data[:, 1]

    def overlapping_frames(self):
        """
        Generates overlapping frames

        Generator that yields a deque containing the current frame of audio
        data.  The deque contents is shifted by the frame size minus the
        overlap on each iteration, to minimise computation.
        """
        frame = collections.deque(maxlen=self.frame_size)
        frame.extend(self.data[:self.frame_size])
        yield frame

        for i in range(1, self.num_frames):
            frame.extend(self.data[i * self.frame_size - self.overlap:
                                   (i + 1) * self.frame_size - self.overlap])
            yield frame
