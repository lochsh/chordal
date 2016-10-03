import hypothesis
from hypothesis.extra.numpy import arrays
import numpy as np

import chordal


def mock_wavfile_read(*args):
    return 800, np.array(range(100))

chordal.AudioProcessor.read_wavfile = mock_wavfile_read


class TestOverlappingFrames:

    def setup_method(self):
        self.ap = chordal.AudioProcessor('')

    @hypothesis.given(arrays(float, 100))
    def test_overlapping_frames_yields_correct_initial_frame(self, data):
        self.ap.data = np.nan_to_num(data)
        self.ap.process_data()

        frames = self.ap.overlapping_frames()
        assert (next(frames) == self.ap.data[:self.ap.frame_size]).all()

    @hypothesis.given(arrays(float, 100))
    def test_overlapping_frames_advances_correctly(self, data):
        self.ap.data = np.nan_to_num(data)
        self.ap.process_data()

        frames = self.ap.overlapping_frames()
        next(frames)
        assert (next(frames) ==
                self.ap.data[self.ap.frame_size - self.ap.overlap:
                             2 * self.ap.frame_size - self.ap.overlap]).all()


def test_ref_freqs():
    """Compare calculated semitone frequencies with a reference list"""
    ref_freqs = [27.5000, 29.1352, 30.8677, 32.7032,
                 34.6478, 36.7081, 38.8909, 41.2034,
                 43.6535, 46.2493, 48.9994, 51.9131]
    assert np.allclose(chordal.Chromagrammer.ref_freqs, ref_freqs)


@hypothesis.given(hypothesis.strategies.floats(min_value=0, max_value=10000))
def test_chromagram_is_constant_for_sine_wave(freq):
    """Test chromagram varies minimally across time shifts for pure sine"""
    ap = chordal.AudioProcessor('')
    ap.data = np.array([np.sin(2*np.pi * x * freq)
                        for x in np.linspace(0, 100, 0.1)])
    ap.process_data()

    frames = ap.overlapping_frames()
    chromagrammer = chordal.Chromagrammer(ap.f_s, 2048)
    chromagram = chromagrammer.chromagram(frames)

    def is_close():
        for _ in range(100):
            yield np.allclose(next(chromagram), next(chromagram), atol=0.1)
    assert all(is_close())


@hypothesis.given(hypothesis.strategies.integers(min_value=100,
                                                 max_value=10000))
def test_chroma_intensity_zero_for_zero_data(length):
    chromagrammer = chordal.Chromagrammer(44100, 2048)
    assert all(chromagrammer.chroma_intensity(np.zeros(length), n) == 0
               for n in range(12))


def test_spectrum_bin_to_chroma_index_zero_when_fbin_is_fref():
    chromagrammer = chordal.Chromagrammer(44100, 2048)
    k = chromagrammer.N * chordal.Chromagrammer.ref_freqs[0]/chromagrammer.f_s
    assert chromagrammer.spectrum_bin_to_chroma_index(k) == 0
