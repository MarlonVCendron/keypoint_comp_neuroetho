import keypoint_moseq as kpms
from utils import load_data_and_config, print_legal
from os import path


def fit_arhmm(
    project_dir,
    model_name,
    num_ar_iters_checkpoint,
    iters,
    kappa_val=None,
    config_overrides=None,
):
    """Ajusta o modelo AR-HMM, carregando de um checkpoint do modelo AR."""
    print_legal(
        f"Iniciando ajuste do AR-HMM para o projeto: {project_dir}, nome do modelo: {model_name}"
    )

    _, _, config = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    # Carrega o modelo a partir do checkpoint do modelo AR
    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir,
        model_name,
        iteration=num_ar_iters_checkpoint,  # Especifica a iteração do modelo AR a ser carregada
    )

    # Modifica kappa. Se não fornecido como argumento, usa o valor da configuração ou o valor padrão.
    kappa_to_use = kappa_val if kappa_val is not None else config.get("kappa", 1e4)
    model = kpms.update_hypparams(model, kappa=kappa_to_use)
    print_legal(f"Usando kappa: {kappa_to_use}")

    # Ajusta o modelo
    print_legal(
        f"Iniciando ajuste do AR-HMM a partir da iteração {current_iter} até {current_iter + iters}."
    )
    model, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=False,
        start_iter=current_iter,
        num_iters=current_iter + iters,
    )
    print_legal(f"Ajuste do AR-HMM completo. Modelo atualizado como {model_name}.")
