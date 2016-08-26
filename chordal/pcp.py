import collections
import numpy as np
from scipy.io import wavfile


class ChordRecogniser:
    ref_freqs = {n: 2**(n/12) * 440.0 for n in range(12)}

    def __init__(self, sampling_freq, fft_length):
        self.f_s = sampling_freq
        self.N = fft_length

    def fft_bin_to_freq(self, bin_count):
        return self.f_s * bin_count / self.N

    def spectrum_bin_to_pcp_index(self, l, f_ref):
        return np.floor(12 * np.log2(self.fft_bin_to_freq(l) / f_ref) % 12)

    def single_pcp(self, data, pcp_index):
        mapping = (abs(np.fft.fft(data))**2
                   for k in range(1, int(self.N/2 - 1))
                   if self.spectrum_bin_to_pcp_index(
                       k, ChordRecogniser.ref_freqs[pcp_index]) == pcp_index)
        return sum(sum(mapping))

    def full_pcp(self, data_frames):
        for frame in data_frames:
            pcp = np.array([self.single_pcp(frame, p)
                            for p in ChordRecogniser.ref_freqs])
            yield pcp/pcp.max()


class AudioProcessor:

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
        frame = collections.deque(maxlen=self.frame_size)
        frame.extend(self.data[:self.frame_size])
        yield frame

        for i in range(1, self.num_frames):
            frame.extend(self.data[i * self.frame_size - self.overlap:
                                   (i + 1) * self.frame_size - self.overlap])
            yield frame
