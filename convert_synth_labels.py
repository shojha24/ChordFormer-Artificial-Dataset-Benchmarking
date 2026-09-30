import os
import json
import re
from pathlib import Path
import complex_chord

AUDIO_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-conditions/naturalistic/audio"
POP_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-target-500k/pop-rock-labels"
JAZZ_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-target-500k/jazz-labels"
OUT_LAB_DIR = "/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking/data/synth_naturalistic_lab"

def normalize_harte(label):
    if not label or label in ('N', 'X'):
        return label
    # Merge consecutive paren groups like (b7)(#9) -> (b7,#9)
    label = re.sub(r'\)\(', ',', label)
    return label

def convert_all():
    os.makedirs(OUT_LAB_DIR, exist_ok=True)
    total_songs = 5000
    converted = 0
    total_chords = 0
    unparseable_chords = 0

    print(f"Converting {total_songs} synthetic songs to .lab format in {OUT_LAB_DIR}...")
    for i in range(total_songs):
        song_name = f"song_{i}"
        if i < 2500:
            json_path = os.path.join(POP_DIR, f"song_{i}.json")
        else:
            json_path = os.path.join(JAZZ_DIR, f"song_{i - 2500}.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing JSON: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        out_lab_path = os.path.join(OUT_LAB_DIR, f"{song_name}.lab")
        with open(out_lab_path, "w") as out:
            for chord in data["chords"]:
                total_chords += 1
                if chord.get("is_no_chord", False):
                    label = "N"
                else:
                    label = normalize_harte(chord["harte"])

                # Sanity check with complex_chord
                try:
                    complex_chord.Chord(label)
                except Exception as e:
                    unparseable_chords += 1
                    print(f"Warning: Song {song_name} unparseable chord '{label}': {e}. Falling back to N")
                    label = "N"

                out.write(f"{chord['time_start']:.3f}\t{chord['time_end']:.3f}\t{label}\n")

        converted += 1
        if (i + 1) % 1000 == 0:
            print(f"Progress: {converted}/{total_songs} songs converted ({total_chords} chords, {unparseable_chords} errors).")

    print(f"Successfully generated {converted} .lab files in {OUT_LAB_DIR}")
    print(f"Total chords checked: {total_chords}, unparseable fallback: {unparseable_chords}")

if __name__ == '__main__':
    convert_all()
