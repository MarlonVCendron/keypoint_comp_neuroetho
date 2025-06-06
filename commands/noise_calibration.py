import keypoint_moseq as kpms
from utils.print_legal import print_legal
from utils.load_data_and_config import load_keypoints, load_config


def noise_calibration(project_dir):
    """Calibra o ruído do modelo em um widget interativo."""

    config = load_config(project_dir)
    coordinates, confidences, bodyparts = load_keypoints(project_dir)

    print_legal(f"Iniciando calibração de ruído para o projeto: {project_dir}")

    kpms.noise_calibration(project_dir, coordinates, confidences, **config)