import heapq
import glob
import os

import chordal


sounds_dir = 'sounds'
chords_dir = os.path.join(sounds_dir, 'chords')


def test_semitones():
    semitones = sorted(glob.glob(os.path.join(sounds_dir, '*5.wav')))

    for note_file, i in zip(semitones, range(len(semitones))):
        chromagram = next(chordal.chromagram(note_file))
        assert chromagram.argmax() == i % 12


def test_major_chords():
    major_chords = sorted(glob.glob(os.path.join(chords_dir, '*major*')))

    for chord_file, i in zip(major_chords, range(len(major_chords))):
        chromagram = next(chordal.chromagram(chord_file))
        chroma = sorted(heapq.nlargest(3, range(len(chromagram)),
                        chromagram.take))
        assert chroma == sorted([i % 12, (i + 4) % 12, (i + 7) % 12])
