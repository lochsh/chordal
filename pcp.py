import numpy as np


class ChordRecogniser:

    def __init__(self, sampling_freq, fft_length):
        self.f_s = sampling_freq
        self.N = fft_length

    def fft_bin_to_freq(self, bin_count):
        return self.f_s * bin_count / self.N

    def spectrum_bin_to_pcp_index(self, l, f_ref):
        round(12 * np.log2(self.fft_bin_to_freq(l) / f_ref) % 12)

    def pcp(self, pcp_index, f_ref):
        return sum(np.fft(l) for l in range(1, self.N/2 - 1)
                   if self.spectrum_bin_to_pcp_index(l, f_ref) == pcp_index)
