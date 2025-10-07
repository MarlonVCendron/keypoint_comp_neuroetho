import keypoint_moseq as kpms  # type: ignore
import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from utils.load_data_and_config import load_config
from utils.print_legal import print_legal
from utils.video_frame_indexes import skip_frames_in_array, start_frames_to_skip, end_frames_to_skip

rearing_csvs_path = "./data/rearing_csv"


def validation(project_dir, model_name):
    results = kpms.load_results(project_dir, model_name)

    all_syllables_rearing = []
    all_syllables_non_rearing = []

    config = load_config(project_dir)
    video_dirs = config["video_dir"]

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

        video_path = get_video_path(recording_name, video_dirs)
        if video_path is not None:
            generate_syllable_rearing_movies(
                syllable, rearing, video_path, recording_name, project_dir, model_name
            )

    analyze_overall_syllable_patterns(all_syllables_rearing, all_syllables_non_rearing)


def get_video_path(recording_name, video_dirs):
    for video_dir in video_dirs:
        for file in os.listdir(video_dir):
            if file.endswith(".mp4"):
                video_name = file.split(".")[0]
                if video_name in recording_name:
                    return os.path.join(video_dir, file)
    return None


def find_csv_file(recording_name):
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


def generate_syllable_rearing_movies(
    syllable, rearing, video_path, recording_name, project_dir, model_name
):
    output_dir = Path(project_dir) / model_name / "syllable_rearing_movies"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{recording_name}_syllable_rearing.mp4"

    print_legal(f"Gerando vídeo: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_legal(f"Erro: Não foi possível abrir o vídeo {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(start_frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            print_legal(f"Erro: Não foi possível pular {start_frames_to_skip} frames iniciais")
            cap.release()
            return

    min_length = min(len(syllable), len(rearing), total_frames - start_frames_to_skip)
    syllable = syllable[:min_length]
    rearing = rearing[:min_length]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    rearing_color = (0, 0, 255)
    non_rearing_color = (0, 255, 0)
    text_color = (255, 255, 255)
    background_color = (0, 0, 0)

    frame_idx = 0
    while frame_idx < min_length:
        ret, frame = cap.read()
        if not ret:
            break

        current_syllable = syllable[frame_idx]
        current_rearing = rearing[frame_idx]

        status_color = rearing_color if current_rearing else non_rearing_color
        status_text = "REARING" if current_rearing else "NORMAL"

        cv2.rectangle(frame, (10, 10), (300, 80), background_color, -1)
        cv2.rectangle(frame, (10, 10), (300, 80), status_color, 3)

        cv2.putText(
            frame, f"Frame: {frame_idx}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2
        )
        cv2.putText(
            frame,
            f"Syllable: {current_syllable}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )
        cv2.putText(frame, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(
            frame,
            f"{frame_idx + 1}/{min_length}",
            (width - 150, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print_legal(f"Vídeo salvo: {output_path}")
