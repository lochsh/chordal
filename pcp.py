import numpy as np


def fft_bin_to_freq(f_s, bin_count, ft_length):
    return f_s * bin_count / ft_length


def spectrum_bin_to_pcp_index(l, f_ref, f_s, N):
    p = round(12 * np.log2((f_s * l) / (N * f_ref)) % 12) if l != 0 else -1
    return p


def pcp(pcp_index, N):
    return sum(np.fft(l) for l in range(1, N/2 - 1)
               if spectrum_bin_to_pcp_index(l) == pcp_index)
