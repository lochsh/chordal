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
    ref_freqs = [440.000, 466.164, 493.883, 523.251, 554.365,
                 587.330, 622.254, 659.255, 698.456, 739.989,
                 783.991, 830.609]
    assert np.allclose(list(chordal.PcpCalculator.ref_freqs.values()),
                       ref_freqs)


def test_full_pcp_is_not_always_the_same():
    ap = chordal.AudioProcessor('')
    frames = ap.overlapping_frames()
    pcp_calc = chordal.PcpCalculator(ap.f_s, 2048)
    pcp = pcp_calc.full_pcp(frames)
    for _ in range(10):
        assert (next(pcp) != next(pcp)).all()


def test_single_pcp_zero_for_zero_data():
    pcp_calc = chordal.PcpCalculator(44100, 2048)
    for n in pcp_calc.ref_freqs:
        assert pcp_calc.single_pcp(np.zeros(100), n) == 0
