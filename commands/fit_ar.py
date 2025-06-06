import keypoint_moseq as kpms
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal
from os import path


def fit_ar(project_dir, model_name, num_ar_iters, config_overrides=None):
    """Ajusta o modelo AR inicial."""

    print_legal(
        f"Ajustando o modelo AR para o projeto: {project_dir}, nome do modelo: {model_name}, iterações: {num_ar_iters}"
    )

    data, metadata, config = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    pca = kpms.load_pca(project_dir)

    model = kpms.init_model(data, pca=pca, **config)

    print_legal(f"Iniciando ajuste do modelo AR com {num_ar_iters} iterações.")

    model, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=True,
        num_iters=num_ar_iters,
    )

    print_legal(f"Ajuste do modelo AR completo. Modelo salvo como {model_name}.")
