import keypoint_moseq as kpms
from utils.print_legal import print_legal


def load_config(project_dir_path):
    """Carrega a configuração do projeto."""
    return kpms.load_config(project_dir_path)

def load_keypoints(project_dir_path):
    """Carrega os keypoints do DLC."""
    config = load_config(project_dir_path)
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        config["video_dir"], "deeplabcut", extension=".csv"
    )
    return coordinates, confidences, bodyparts

def load_data_and_config(project_dir_path):
    """Carrega a configuração e os keypoints do DLC."""
    print_legal(f"Carregando configuração e keypoints do DLC de: {project_dir_path}")
    config = load_config(project_dir_path)
    coordinates, confidences, bodyparts = load_keypoints(project_dir_path)
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    return data, metadata, config
