import keypoint_moseq as kpms
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal
from utils.args import project_dir
from os import path


def fit_pca(project_dir_path, config_overrides=None):
    """Fits and saves PCA."""

    data, metadata, config = load_data_and_config(project_dir_path)
    if config_overrides:
        config.update(config_overrides)

    print_legal(f"Ajustando PCA para o projeto: {project_dir_path}")

    pca = kpms.fit_pca(**data, **config)
    kpms.save_pca(pca, project_dir)

    kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=project_dir)
    kpms.plot_pcs(pca, project_dir=project_dir, **config)

    print_legal(f"PCA ajustada e salva em {path.join(project_dir_path, 'pca.p')}")
