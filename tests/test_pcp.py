import numpy as np

import chordal

mock_data = np.array(range(10000))


def mock_wavfile_read(*args):
    return 44100, mock_data

chordal.AudioProcessor.read_wavfile = mock_wavfile_read


class TestOverlappingFrames:
    ap = chordal.AudioProcessor('')

    def test_overlapping_frames_yields_correct_initial_frame(self):
        frames = self.ap.overlapping_frames()
        assert (next(frames) == mock_data[:self.ap.frame_size]).all()

    def test_overlapping_frames_advances_correctly(self):
        frames = self.ap.overlapping_frames()
        next(frames)
        assert (next(frames) ==
                mock_data[self.ap.frame_size - self.ap.overlap:
                          2 * self.ap.frame_size - self.ap.overlap]).all()


def test_ref_freqs():
    """Compare calculated semitone frequencies with a reference list"""
    ref_freqs = [27.5000, 29.1352, 30.8677, 32.7032,
                 34.6478, 36.7081, 38.8909, 41.2034,
                 43.6535, 46.2493, 48.9994, 51.9131]
    assert np.allclose(chordal.Chromagrammer.ref_freqs,
                       ref_freqs)


def test_chromagram_is_not_always_the_same():
    ap = chordal.AudioProcessor('')
    frames = ap.overlapping_frames()
    chromagrammer = chordal.Chromagrammer(ap.f_s, 2048)
    chromagram = chromagrammer.chromagram(frames)

    def is_same():
        for _ in range(100):
            yield (next(chromagram) == next(chromagram)).all()
    assert not all(is_same())


def test_chroma_intensity_zero_for_zero_data():
    chromagrammer = chordal.Chromagrammer(44100, 2048)
    for n in range(12):
        assert chromagrammer.chroma_intensity(np.zeros(100), n) == 0
