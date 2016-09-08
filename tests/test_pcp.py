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
    ref_freqs = [440.000, 466.164, 493.883, 523.251, 554.365,
                 587.330, 622.254, 659.255, 698.456, 739.989,
                 783.991, 830.609]
    assert np.allclose(list(chordal.Chromagrammer.ref_freqs.values()),
                       ref_freqs)


def test_full_chroma_is_not_always_the_same():
    ap = chordal.AudioProcessor('')
    frames = ap.overlapping_frames()
    chroma_calc = chordal.Chromagrammer(ap.f_s, 2048)
    chroma = chroma_calc.full_chroma(frames)

    def is_same():
        for _ in range(100):
            yield (next(chroma) == next(chroma)).all()
    assert not all(is_same())


def test_single_chroma_zero_for_zero_data():
    chroma_calc = chordal.Chromagrammer(44100, 2048)
    for n in chroma_calc.ref_freqs:
        assert chroma_calc.single_chroma(np.zeros(100), n) == 0
