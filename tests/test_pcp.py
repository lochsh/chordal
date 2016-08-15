import numpy as np

import chordal

mock_data = np.array(range(10000))


def mock_wavfile_read(*args):
    return 44100, mock_data

chordal.pcp.AudioProcessor.read_wavfile = mock_wavfile_read


class TestOverlappingFrames:
    ap = chordal.pcp.AudioProcessor('')

    def test_overlapping_frames_yields_correct_initial_frame(self):
        frames = self.ap.overlapping_frames()
        assert (next(frames) == mock_data[:self.ap.frame_size]).all()

    def test_overlapping_frames_advances_correctly(self):
        frames = self.ap.overlapping_frames()
        next(frames)
        assert (next(frames) ==
                mock_data[self.ap.frame_size - self.ap.overlap:
                          2*self.ap.frame_size - self.ap.overlap]).all()

if __name__ == '__main__':
    t = TestOverlappingFrames()
    frames = t.ap.overlapping_frames()
    for _ in range(10):
        print(next(frames))
