import keypoint_moseq as kpms  # type: ignore
import h5py
import os

from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal
from utils.overwrite_results import prevent_overwrite_error


def apply(project_dir, model_name):
    model = kpms.load_checkpoint(project_dir, model_name)[0]

    # # Teste
    # new_data = ["./projects/elm_ms/data/vids/T"]

    # Sample
    new_data = ['./projects/elm_ms/data/vids/S1', './projects/elm_ms/data/vids/S2']

    data, metadata, config, coordinates, video_frame_indexes, _ = load_data_and_config(
        project_dir, video_dir=new_data
    )

    prevent_overwrite_error(project_dir, model_name, coordinates.keys())

    results = kpms.apply_model(
        model, data, metadata, project_dir, model_name, parallel_message_passing=False, **config
    )

    kpms.save_results_as_csv(results, project_dir, model_name)

