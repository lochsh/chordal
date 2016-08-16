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


def test_ref_frequencies():
    ref_frequencies = [440.000, 466.164, 493.883, 523.251, 554.365,
                       587.330, 622.254, 659.255, 698.456, 739.989,
                       783.991, 830.609]
    assert np.allclose(list(chordal.ChordRecogniser.ref_frequencies.values()),
                       ref_frequencies)
