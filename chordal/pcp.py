import collections
import numpy as np
from scipy.io import wavfile


class Chromagrammer:
    """
    Calculates chromagrams, a.k.a. Pitch Class Profile (PCP) vectors

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
        """
        Calculates the PCP level for a particular pitch for a single sample.

        Calculates the PCP level for a given pitch, defined by the pcp_index
        parameter, for a single time-domain sample.  Effectively, we are
        summing the components of the DFT that correspond to this pitch.
        """
        mapping = (abs(np.fft.fft(data))**2
                   for k in range(1, int(self.N/2 - 1))
                   if self.spectrum_bin_to_pcp_index(
                       k, Chromagrammer.ref_freqs[pcp_index]) == pcp_index)
        return sum(sum(mapping))

    def chromagram(self, data_frames):
        """Calculate chromagrams for successive time-domain samples"""
        for frame in data_frames:
            chroma = np.array([self.single_chroma(frame, c)
                               for c in Chromagrammer.ref_freqs])
            yield chroma/chroma.max()


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
