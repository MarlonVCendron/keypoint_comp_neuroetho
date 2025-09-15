import keypoint_moseq as kpms # type: ignore
from utils.print_legal import print_legal


def load_config(project_dir):
    """Carrega a configuração do projeto."""
    return kpms.load_config(project_dir)

def load_keypoints(project_dir):
    """Carrega os keypoints do DLC."""
    config = load_config(project_dir)
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        config["video_dir"], "deeplabcut", extension=".csv"
    )
    return coordinates, confidences, bodyparts

def load_data_and_config(project_dir):
    """Carrega a configuração e os keypoints do DLC."""
    print_legal(f"Carregando configuração e keypoints do DLC de: {project_dir}")
    config = load_config(project_dir)
    coordinates, confidences, bodyparts = load_keypoints(project_dir)
    coordinates, confidences, video_frame_indexes = video_frame_indexes(coordinates, confidences)
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    return data, metadata, config, coordinates, video_frame_indexes
