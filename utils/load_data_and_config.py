import keypoint_moseq as kpms # type: ignore
from utils.print_legal import print_legal
from utils.video_frame_indexes import get_video_frame_indexes
from utils.find_medoid_distance_outliers import filter_outliers

def load_config(project_dir):
    """Carrega a configuração do projeto."""
    return kpms.load_config(project_dir)

def load_keypoints(project_dir, video_dir=None):
    """Carrega os keypoints do DLC."""
    config = load_config(project_dir)
    if video_dir is None:
        video_dir = config["video_dir"]
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        video_dir, "deeplabcut", extension=".csv"
    )
    return coordinates, confidences, bodyparts

def load_data_and_config(project_dir, remove_outliers=True, video_dir=None):
    """Carrega a configuração e os keypoints do DLC."""
    print_legal(f"Carregando configuração e keypoints do DLC de: {project_dir}")

    config = load_config(project_dir)
    coordinates, confidences, bodyparts = load_keypoints(project_dir, video_dir)
    coordinates, confidences, video_frame_indexes = get_video_frame_indexes(coordinates, confidences)
    if remove_outliers:
        coordinates, confidences, _ = filter_outliers(coordinates, confidences, config, project_dir=project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    return data, metadata, config, coordinates, video_frame_indexes, confidences
