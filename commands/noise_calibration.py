import keypoint_moseq as kpms
from utils.print_legal import print_legal
from utils.load_data_and_config import load_keypoints, load_config


def noise_calibration(project_dir_path):
    """Calibra o ruído do modelo em um widget interativo."""

    config = load_config(project_dir_path)
    coordinates, confidences, bodyparts = load_keypoints(project_dir_path)

    print_legal(f"Iniciando calibração de ruído para o projeto: {project_dir_path}")

    kpms.noise_calibration(project_dir_path, coordinates, confidences, **config)