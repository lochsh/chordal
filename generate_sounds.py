import subprocess

import chordal

chroma_names = ['A', 'B_flat', 'B', 'C', 'C_sharp', 'D',
                'E_flat', 'E', 'F', 'F_sharp', 'G', 'A_flat']
for freq, chroma, octave in zip(chordal.Chromagrammer.ref_freqs,
                                chroma_names, range(7)):
    subprocess.call('ffmpeg -f lavfi -i '
                    'sine=frequency={0}:duration=5 {1}_{2}.wav'
                    .format(freq, chroma, octave).split())
