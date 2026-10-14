import os
import json
import re
import argparse
from pathlib import Path
import complex_chord

DEFAULT_AUDIO_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-conditions-10k/naturalistic/audio"
DEFAULT_POP_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-target-10k/pop-rock-labels"
DEFAULT_JAZZ_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-target-10k/jazz-labels"
DEFAULT_OUT_LAB_DIR = "/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking/data/synth_naturalistic_lab"

def normalize_harte(label):
    if not label or label in ('N', 'X'):
        return label
    # Merge consecutive paren groups like (b7)(#9) -> (b7,#9)
    label = re.sub(r'\)\(', ',', label)
    return label

def convert_all(total_songs=10000, pop_count=6500, pop_dir=DEFAULT_POP_DIR, jazz_dir=DEFAULT_JAZZ_DIR, out_dir=DEFAULT_OUT_LAB_DIR):
    os.makedirs(out_dir, exist_ok=True)
    converted = 0
    total_chords = 0
    unparseable_chords = 0

    print(f"Converting {total_songs} synthetic songs ({pop_count} Pop/Rock, {total_songs - pop_count} Jazz) to .lab format in {out_dir}...")
    for i in range(total_songs):
        song_name = f"song_{i}"
        if i < pop_count:
            json_path = os.path.join(pop_dir, f"song_{i}.json")
        else:
            json_path = os.path.join(jazz_dir, f"song_{i - pop_count}.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing JSON: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        out_lab_path = os.path.join(out_dir, f"{song_name}.lab")
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

    print(f"Successfully generated {converted} .lab files in {out_dir}")
    print(f"Total chords checked: {total_chords}, unparseable fallback: {unparseable_chords}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert synthetic JSON annotations to Harte .lab format")
    parser.add_argument('--total_songs', type=int, default=10000, help="Total songs (default: 10000)")
    parser.add_argument('--pop_count', type=int, default=6500, help="Number of pop/rock songs (default: 6500)")
    parser.add_argument('--pop_dir', type=str, default=DEFAULT_POP_DIR, help="Directory containing pop/rock JSONs")
    parser.add_argument('--jazz_dir', type=str, default=DEFAULT_JAZZ_DIR, help="Directory containing jazz JSONs")
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_LAB_DIR, help="Output directory for .lab files")
    args = parser.parse_args()

    convert_all(
        total_songs=args.total_songs,
        pop_count=args.pop_count,
        pop_dir=args.pop_dir,
        jazz_dir=args.jazz_dir,
        out_dir=args.out_dir,
    )
