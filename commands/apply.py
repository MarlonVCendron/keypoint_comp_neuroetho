import keypoint_moseq as kpms  # type: ignore
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal


def apply(project_dir, model_name):
    model = kpms.load_checkpoint(project_dir, model_name)[0]

    new_data = [
        './projects/elm_ms/data/vids/G0',
        './projects/elm_ms/data/vids/G39',
        './projects/elm_ms/data/vids/G41',
        './projects/elm_ms/data/vids/G42',
        './projects/elm_ms/data/vids/G43',
        './projects/elm_ms/data/vids/G45',
        './projects/elm_ms/data/vids/G64',
        './projects/elm_ms/data/vids/G65',
        './projects/elm_ms/data/vids/G66',
        './projects/elm_ms/data/vids/G67',
        './projects/elm_ms/data/vids/G68',
    ]

    data, metadata, config, coordinates, video_frame_indexes, _ = load_data_and_config(project_dir, video_dir=new_data)

    results = kpms.apply_model(model, data, metadata, project_dir, model_name, parallel_message_passing=False, **config)

    kpms.save_results_as_csv(results, project_dir, model_name)
