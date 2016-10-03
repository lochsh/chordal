import os
import shlex
import subprocess

import chordal

chroma_names = ['A', 'A_sharp', 'B', 'C', 'C_sharp', 'D',
                'D_sharp', 'E', 'F', 'F_sharp', 'G', 'G_sharp']


def create_semitones(output_dir):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    for octave in range(2, 7):
        for freq, chroma in zip(chordal.Chromagrammer.ref_freqs, chroma_names):
            subprocess.call(shlex.split('ffmpeg -f lavfi -i '
                                        '"sine=frequency={0}:duration=5" '
                                        '{1}/{2}_{3}.wav'
                                        .format(freq * (2 ** octave),
                                                output_dir, octave, chroma)))


def create_chord(tone_files, output_file):
    subprocess.call(shlex.split('ffmpeg -i {0} -filter_complex '
                                '"[0:a][1:a][2:a]amerge=inputs=3[aout]" '
                                '-map "[aout]" {1}'
                                .format(' -i '.join(tone_files), output_file)))


def create_chords(input_dir, output_dir, major=True, octave=5):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    third, name = (4, 'major') if major else (3, 'minor')

    for i in range(12):
        chord = [chroma_names[i % 12],
                 chroma_names[(i+third) % 12],
                 chroma_names[(i+7) % 12]]
        create_chord([os.path.join(input_dir, '{}_{}.wav'.format(octave, note))
                      for note in chord],
                     os.path.join(output_dir, '{}_{}.wav'
                                              .format(chord[0], name)))


if __name__ == '__main__':
    output_dir = 'sounds'
    create_semitones(output_dir)
    create_chords(output_dir, os.path.join(output_dir, 'chords'))
    create_chords(output_dir, os.path.join(output_dir, 'chords'), major=False)
