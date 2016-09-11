import os
import shlex
import subprocess

import chordal

chroma_names = ['A', 'B_flat', 'B', 'C', 'C_sharp', 'D',
                'E_flat', 'E', 'F', 'F_sharp', 'G', 'A_flat']


def create_semitones(output_dir):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    for octave in range(8):
        for freq, chroma in zip(chordal.Chromagrammer.ref_freqs, chroma_names):
            subprocess.call(shlex.split('ffmpeg -f lavfi -i '
                                        '"sine=frequency={0}:duration=5" '
                                        '{1}/{2}_{3}.wav'
                                        .format(freq * (octave + 1),
                                                output_dir, chroma, octave)))


def create_chord(tone_files, output_file):
    subprocess.call(shlex.split('ffmpeg -i {0} -filter_complex '
                                '"[0:a][1:a]amerge=inputs=2[aout]" '
                                '-map "[aout]" {1}'
                                .format(' -i '.join(tone_files), output_file)))


if __name__ == '__main__':
    output_dir = 'sounds'
    create_semitones(output_dir)
    a_minor = ['A', 'C', 'E']
    create_chord(['{0}/{1}_5.wav'.format(output_dir, chroma)
                  for chroma in a_minor], os.path.join(output_dir, 'am.wav'))
