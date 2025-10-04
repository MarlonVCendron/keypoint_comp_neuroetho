import keypoint_moseq as kpms  # type: ignore
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal


def validation(
    project_dir,
    model_name,
    checkpoint,
    config_overrides=None,
):
    _, _, config, coordinates, video_frame_indexes, _ = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir,
        model_name,
        iteration=checkpoint,
    )

    results = kpms.apply_model(model, data, metadata, project_dir, model_name, **config)

    print(results)


