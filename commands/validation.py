import keypoint_moseq as kpms  # type: ignore
import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from utils.load_data_and_config import load_config
from utils.print_legal import print_legal
from utils.video_frame_indexes import skip_frames_in_array, start_frames_to_skip, end_frames_to_skip

rearing_csvs_path = "./data/rearing_csv"
generate_movies = False

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

        if generate_movies:
            video_path = get_video_path(recording_name, video_dirs)
            if video_path is not None:
                generate_syllable_rearing_movies(
                    syllable, rearing, video_path, recording_name, project_dir, model_name
                )

    analyze_syllable_patterns(all_syllables_rearing, all_syllables_non_rearing)
    
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


def analyze_syllable_patterns(syllables_rearing, syllables_non_rearing):
    if not syllables_rearing:
        print("Nenhum evento de rearing encontrado")
        return

    print_legal(f"ANÁLISE PREDITIVA DE SÍLABAS")
    print(f"Total de frames com rearing: {len(syllables_rearing)}")
    print(f"Total de frames sem rearing: {len(syllables_non_rearing)}")
    
    y_true = np.concatenate([np.ones(len(syllables_rearing)), np.zeros(len(syllables_non_rearing))])
    all_syllables = syllables_rearing + syllables_non_rearing
    
    syllable_counts = Counter(all_syllables)
    frequent_syllables = [syl for syl, count in syllable_counts.items() if count >= 10]
    
    syllable_metrics = {}
    
    for syllable in frequent_syllables:
        y_pred = np.array([1 if syl == syllable else 0 for syl in all_syllables])
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        syllable_metrics[syllable] = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1_score': f1,
            'accuracy': accuracy, 'total_occurrences': tp + fp
        }
    
    sorted_syllables = sorted(syllable_metrics.items(), 
                            key=lambda x: x[1]['f1_score'], reverse=True)
    
    print(f"\nTop sílabas preditivas de rearing (ordenadas por F1-score):")
    print("Sílaba | TP | TN | FP | FN | Precision | Recall | F1-score | Accuracy | Total")
    print("-" * 85)
    
    for i, (syllable, metrics) in enumerate(sorted_syllables[:15]):
        print(f"{syllable:6d} | {metrics['tp']:2d} | {metrics['tn']:4d} | {metrics['fp']:2d} | {metrics['fn']:2d} | "
              f"{metrics['precision']:8.3f} | {metrics['recall']:6.3f} | {metrics['f1_score']:8.3f} | "
              f"{metrics['accuracy']:8.3f} | {metrics['total_occurrences']:5d}")
    
    if sorted_syllables:
        best_syllable = sorted_syllables[0]
        print(f"\nMelhor sílaba preditiva: {best_syllable[0]}")
        print(f"- F1-score: {best_syllable[1]['f1_score']:.3f}")
        print(f"- Precision: {best_syllable[1]['precision']:.3f}")
        print(f"- Recall: {best_syllable[1]['recall']:.3f}")
        print(f"- Accuracy: {best_syllable[1]['accuracy']:.3f}")


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
