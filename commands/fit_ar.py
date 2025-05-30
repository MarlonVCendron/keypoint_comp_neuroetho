import keypoint_moseq as kpms
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal
from os import path


def fit_ar(project_dir_path, model_name_str, num_ar_iters_val, config_overrides=None):
    """Ajusta o modelo AR inicial."""

    print_legal(
        f"Ajustando o modelo AR para o projeto: {project_dir_path}, nome do modelo: {model_name_str}, iterações: {num_ar_iters_val}"
    )

    data, metadata, config = load_data_and_config(project_dir_path)
    if config_overrides:
        config.update(config_overrides)

    pca = kpms.load_pca(project_dir_path)

    model = kpms.init_model(data, pca=pca, **config)

    print_legal(f"Iniciando ajuste do modelo AR com {num_ar_iters_val} iterações.")

    model, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir_path,
        model_name_str,
        ar_only=True,
        num_iters=num_ar_iters_val,
    )

    print_legal(f"Ajuste do modelo AR completo. Modelo salvo como {model_name_str}.")
