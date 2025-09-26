import keypoint_moseq as kpms  # type: ignore
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal


def fit_full_model(
    project_dir,
    model_name,
    checkpoint,
    iters,
    kappa=None,
    config_overrides=None,
):
    _, _, config, _, _, _ = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    # Carrega o modelo a partir do checkpoint do modelo AR
    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir,
        model_name,
        iteration=checkpoint,
    )

    print_legal(
        f"Iniciando ajuste do modelo completo para o projeto: {project_dir}, nome do modelo: {model_name}"
    )

    kappa_to_use = kappa if kappa is not None else config.get("kappa", 1e4)
    model = kpms.update_hypparams(model, kappa=kappa_to_use)
    print_legal(f"Usando kappa: {kappa_to_use}")

    model, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=False,
        start_iter=current_iter,
        num_iters=current_iter + iters,
        # parallel_message_passing=False # SALVA USO DE MEMÓRIA https://keypoint-moseq.readthedocs.io/en/latest/FAQs.html#out-of-memor
    )
    print_legal(f"Ajuste do modelo completo {model_name} completo.")
