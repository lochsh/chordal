import hypothesis
from hypothesis.extra.numpy import arrays
import numpy as np

import chordal


def mock_wavfile_read(*args):
    return 44100, np.array(range(100))

chordal.AudioProcessor.read_wavfile = mock_wavfile_read


class TestOverlappingFrames:
    ap = chordal.AudioProcessor('')

    @hypothesis.given(arrays(float, 100))
    def test_overlapping_frames_yields_correct_initial_frame(self, data):
        TestOverlappingFrames.ap.data = data
        TestOverlappingFrames.ap.process_data()

        frames = self.ap.overlapping_frames()
        assert (next(frames) == data[:self.ap.frame_size]).all()

    @hypothesis.given(arrays(float, 100))
    def test_overlapping_frames_advances_correctly(self, data):
        TestOverlappingFrames.ap.data = data
        TestOverlappingFrames.ap.process_data()

        frames = self.ap.overlapping_frames()
        next(frames)
        assert (next(frames) ==
                data[self.ap.frame_size - self.ap.overlap:
                     2 * self.ap.frame_size - self.ap.overlap]).all()


def test_ref_freqs():
    """Compare calculated semitone frequencies with a reference list"""
    ref_freqs = [27.5000, 29.1352, 30.8677, 32.7032,
                 34.6478, 36.7081, 38.8909, 41.2034,
                 43.6535, 46.2493, 48.9994, 51.9131]
    assert np.allclose(chordal.Chromagrammer.ref_freqs, ref_freqs)


@hypothesis.given(arrays(float, 100))
def test_chromagram_is_not_always_the_same(data):
    ap = chordal.AudioProcessor('')
    ap.data = data
    ap.process_data()

    frames = ap.overlapping_frames()
    chromagrammer = chordal.Chromagrammer(ap.f_s, 2048)
    chromagram = chromagrammer.chromagram(frames)

    def is_same():
        for _ in range(100):
            yield (next(chromagram) == next(chromagram)).all()
    assert not all(is_same())


@hypothesis.given(arrays(float, 100))
def test_chromagram_is_not_always_basically_the_same(data):
    ap = chordal.AudioProcessor('')
    ap.data = data
    ap.process_data()

    frames = ap.overlapping_frames()
    chromagrammer = chordal.Chromagrammer(ap.f_s, 2048)
    chromagram = chromagrammer.chromagram(frames)

    def is_close():
        for _ in range(100):
            yield np.allclose(next(chromagram), next(chromagram), atol=0.1)
    assert not all(is_close())


def test_chroma_intensity_zero_for_zero_data():
    chromagrammer = chordal.Chromagrammer(44100, 2048)
    for n in range(12):
        assert chromagrammer.chroma_intensity(np.zeros(100), n) == 0


def test_spectrum_bin_to_chroma_index_zero_when_fbin_is_fref():
    chromagrammer = chordal.Chromagrammer(44100, 2048)
    f_ref = 100
    k = chromagrammer.N * f_ref / chromagrammer.f_s
    assert chromagrammer.spectrum_bin_to_chroma_index(k, f_ref) == 0
