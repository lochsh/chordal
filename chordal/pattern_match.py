import numpy as np
from scipy.spatial import distance


def chord_pattern(chroma_ind, major=True):
    """
    Return theoretical chromagram for major or minor chord

    Returns what should theoretically be the chromagram for a major or minor
    chord, i.e. 1.0 on the 1st, 3rd and 5th, and zero elsewhere
    """
    third = 4 if major else 3
    chromagram = np.zeros(12)
    chromagram[[chroma_ind, (chroma_ind + third) % 12,
                (chroma_ind + 7) % 12]] = 1.0
    return chromagram


def compare_chromagrams(test, major=True):
    """
    Yield chroma index of chord with chromagram nearest to test chromagram

    Yields chroma index of the root of the major or minor chord whose
    chromagram is the shortest Euclidean distance from the test chromagram.
    """
    for t in test:
        d = np.asarray([distance.euclidean(t, chord_pattern(i, major))
                        for i in range(12)])
        yield d.argmin()
