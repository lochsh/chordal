import collections
import logging

import numpy as np
from scipy.io import wavfile

logging.basicConfig(level=logging.DEBUG)


class Chromagrammer:
    """
    Calculates chromagrams, a.k.a. Pitch Class Profile (PCP) vectors

    Calculates chromagrams, 12 dimensional vectors indicating the relative
    intensity of musical semitones across a time-domain signal.  A chroma or
    pitch class refers to a note, e.g. C, regardless of tone height a.k.a
    octave.

    The chromagrams are calculated for a given sampling frequency and FFT
    length.
    """
    ref_freqs = [2**(n/12) * 27.50 for n in range(12)]

    def __init__(self, sampling_freq, fft_length):
        self.f_s = sampling_freq
        self.N = fft_length

    def spectrum_bin_to_chroma_index(self, k):
        """
        Translate spectrum bin index k to nearest chroma index

        Assign a spectrum bin index k to its nearest chroma index, using the
        reference frequency of the 0th chroma as f_ref:

            * The frequency represented by the bin is easily found by the
              product of the sampling frequency and the bin index, divided by
              the DFT length.

            * log_2(f_bin / f_ref) gives how many octaves the bin frequency is
              above the chroma reference frequency

            * 12 * log_2(f_bin / f_ref) gives how many semitones the bin
              frequency is above the chroma reference frequency

            * Rounding this and taking the mod 12 gives the nearest chroma
              index
        """

        def spectrum_bin_to_freq():
            return self.f_s * k / self.N
        
        f_ref = Chromagrammer.ref_freqs[0]
        return round(12 * np.log2(spectrum_bin_to_freq() / f_ref)) % 12

    def chroma_intensity(self, data, chroma_ind):
        """
        Calculates the intensity of a particular pitch in a single time sample.

        Calculates the intensity of a given pitch, defined by the chroma index
        provided, for a single time-domain sample.  Effectively we are summing
        the components of the DFT that correspond to this pitch.
        """
        fft = np.fft.rfft(np.hanning(len(data)) * data, self.N)
        mapping = (abs(fft[k])**2 for k in range(1, int(self.N/2))
                   if self.spectrum_bin_to_chroma_index(k) == chroma_ind)
        return sum(mapping)

    def chromagram(self, data_frames):
        """
        Calculate chromagrams for successive time-domain samples

        Calculates chromagrams for successive time-domain samples, yielding 12
        dimensional vectors for each sample.  Each entry in the vector is the
        relative intensity of a chroma (e.g. the note C), normalised to be
        between 0 and 1.
        """
        for frame in data_frames:
            chromagram = np.array([self.chroma_intensity(frame, c)
                                   for c in range(12)])
            yield chromagram/chromagram.max()


class AudioProcessor:
    """
    Performs audio pre-processing,

    Pre-processes audio from a wavfile, combining the any channels and
    dividing the audio into overlapping frames.
    """

    def __init__(self, file_name, window_len_s=0.025, overlap_s=0.01):
        self.f_s, self.data = self.read_wavfile(file_name)

        self.frame_size = None
        self.overlap = None
        self.num_frames = None
        self.process_data(window_len_s, overlap_s)

    def process_data(self, window_len_s=0.025, overlap_s=0.01):
        self.frame_size = int(self.f_s * window_len_s)
        self.overlap = int(self.f_s * overlap_s)
        self.num_frames = int(len(self.data)/self.overlap)

    @staticmethod
    def read_wavfile(filename):
        f_s, raw_data = wavfile.read(filename)
        try:
            return f_s, raw_data.sum(1)
        except ValueError as e:
            if len(raw_data.shape) == 1:
                return f_s, raw_data
            else:
                logging.error('Provided wav file has unexpected format.')
                raise e

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


def chromagram(wav_file, fft_length=2048):
    audio_processor = AudioProcessor(wav_file)
    frames = audio_processor.overlapping_frames()
    return Chromagrammer(audio_processor.f_s, fft_length).chromagram(frames)
