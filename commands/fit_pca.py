import keypoint_moseq as kpms # type: ignore
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal
from os import path


def fit_pca(project_dir, config_overrides=None):
    """Fits and saves PCA."""

    data, metadata, config, _, _, _ = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    print_legal(f"Ajustando PCA para o projeto: {project_dir}")

    pca = kpms.fit_pca(**data, **config)
    kpms.save_pca(pca, project_dir)

    kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.print_dims_to_explain_variance(pca, 0.95)
    kpms.print_dims_to_explain_variance(pca, 0.99)

    kpms.plot_scree(pca, project_dir=project_dir)
    kpms.plot_pcs(pca, project_dir=project_dir, **config)

    print_legal(f"PCA ajustada e salva em {path.join(project_dir, 'pca.p')}")
    print_legal(
        f"Atualize `latent_dim` em config.yml para o número de PCs a ser usado no modelo (Recomendado usar o que explica 90% da variância).",
        type="warn",
    )
