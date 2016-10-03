import heapq
import glob
import os

import chordal


sounds_dir = 'sounds'
chords_dir = os.path.join(sounds_dir, 'chords')


def test_semitones():
    semitones = sorted(glob.glob(os.path.join(sounds_dir, '*.wav')))

    for note_file, i in zip(semitones, range(len(semitones))):
        print(note_file)
        chromagram = next(chordal.chromagram(note_file))
        assert chromagram.argmax() == i % 12


def _test_chords(major=True):
    third, name = (4, 'major') if major else (3, 'minor')
    chords = sorted(glob.glob(os.path.join(chords_dir, '*{}*'.format(name))))

    for chord_file, i in zip(chords, range(len(chords))):
        chromagram = next(chordal.chromagram(chord_file))
        chroma = sorted(heapq.nlargest(3, range(len(chromagram)),
                        chromagram.take))
        assert chroma == sorted([i % 12, (i + third) % 12, (i + 7) % 12])


def test_major_chords():
    _test_chords()


def test_minor_chords():
    _test_chords(major=False)
