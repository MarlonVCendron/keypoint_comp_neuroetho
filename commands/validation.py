import keypoint_moseq as kpms  # type: ignore
import csv
import os
import numpy as np

from utils.print_legal import print_legal
from utils.video_frame_indexes import skip_frames_in_array

rearing_csvs_path = "./data/rearing_csv"

def validation(project_dir, model_name):
    results = kpms.load_results(project_dir, model_name)
    
    all_syllables_rearing = []
    all_syllables_non_rearing = []
    
    for recording_name, recording in results.items():
        file_path = find_csv_file(recording_name)
        if file_path is None:
            continue

        print_legal(f"Arquivo encontrado: {file_path}")

        syllable = recording["syllable"]
        rearing = get_rearing_data(file_path)

        syllables_rearing, syllables_non_rearing = get_syllables_by_rearing(syllable, rearing)
        all_syllables_rearing.extend(syllables_rearing)
        all_syllables_non_rearing.extend(syllables_non_rearing)
    
    analyze_overall_syllable_patterns(all_syllables_rearing, all_syllables_non_rearing)


def find_csv_file(recording_name):
    print(recording_name, os.listdir(rearing_csvs_path))
    for file in os.listdir(rearing_csvs_path):
        video_name = file.split(".")[0]
        if video_name in recording_name:
            return os.path.join(rearing_csvs_path, file)
    return None

def get_rearing_data(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        rearing_data = []
        for row in reader:
            rearing_data.append(int(row[0]))
        rearing_data = np.array(rearing_data)
    return skip_frames_in_array(rearing_data)

def get_syllables_by_rearing(syllable, rearing):
    assert len(syllable) == len(rearing), "Número de frames de sílabas e rearing não são iguais"
    
    rearing_indices = np.where(rearing == 1)[0]
    non_rearing_indices = np.where(rearing == 0)[0]
    
    syllables_rearing = syllable[rearing_indices].tolist()
    syllables_non_rearing = syllable[non_rearing_indices].tolist()
    
    return syllables_rearing, syllables_non_rearing

def analyze_overall_syllable_patterns(syllables_rearing, syllables_non_rearing):
    if not syllables_rearing:
        print("Nenhum evento de rearing encontrado")
        return
    
    unique_rearing, counts_rearing = np.unique(syllables_rearing, return_counts=True)
    sorted_rearing_idx = np.argsort(counts_rearing)[::-1]
    top_rearing_syllables = unique_rearing[sorted_rearing_idx]
    top_rearing_counts = counts_rearing[sorted_rearing_idx]
    
    unique_non_rearing, counts_non_rearing = np.unique(syllables_non_rearing, return_counts=True)
    sorted_non_rearing_idx = np.argsort(counts_non_rearing)[::-1]
    top_non_rearing_syllables = unique_non_rearing[sorted_non_rearing_idx]
    top_non_rearing_counts = counts_non_rearing[sorted_non_rearing_idx]
    
    print(f"\n=== ANÁLISE GERAL DE SÍLABAS ===")
    print(f"Total de frames com rearing: {len(syllables_rearing)}")
    print(f"Total de frames sem rearing: {len(syllables_non_rearing)}")
    
    print(f"\nTop sílabas DURANTE rearing:")
    for i in range(len(top_rearing_syllables)):
        syl = top_rearing_syllables[i]
        count = top_rearing_counts[i]
        pct = (count / len(syllables_rearing)) * 100
        print(f"  {i+1}. Sílaba {syl}: {count} vezes ({pct:.1f}%)")
    
    print(f"\nTop sílabas SEM rearing:")
    for i in range(len(top_non_rearing_syllables)):
        syl = top_non_rearing_syllables[i]
        count = top_non_rearing_counts[i]
        pct = (count / len(syllables_non_rearing)) * 100
        print(f"  {i+1}. Sílaba {syl}: {count} vezes ({pct:.1f}%)")