import heapq
import glob
import os

import chordal


sounds_dir = 'sounds'
chords_dir = os.path.join(sounds_dir, 'chords')


def test_semitones(octave=5):
    semitones = sorted(glob.glob(os.path.join(sounds_dir,
                                              '*{}.wav'.format(octave))))

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


def _test_chord_pattern_match(major=True):
    third, name = (4, 'major') if major else (3, 'minor')
    chords = sorted(glob.glob(os.path.join(chords_dir, '*{}*'.format(name))))

    for chord_file, i in zip(chords, range(len(chords))):
        chromagram = chordal.chromagram(chord_file)
        comparer = chordal.pattern_match.compare_chromagrams(chromagram, major)
        assert all(next(comparer) == i % 12 for _ in range(100))


def test_major_chords():
    _test_chords()


def test_minor_chords():
    _test_chords(major=False)


def test_major_chords_pattern_match():
    _test_chord_pattern_match()


def test_minor_chords_pattern_match():
    _test_chord_pattern_match(major=False)
