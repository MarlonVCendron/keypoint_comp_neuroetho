import keypoint_moseq as kpms
from utils.print_legal import print_legal
from utils.args import get_subparser
from os import path, makedirs


def init_project(project_dir):
    """Inicializa o projeto."""

    if not project_dir:
        print_legal("Diretório do projeto não informado (--project-dir)", type="error")
        get_subparser("init").print_help()
        return

    kpms.setup_project(project_dir)

    # Cria o diretório `data`
    data_dir_path = path.join(project_dir, 'data')
    makedirs(data_dir_path, exist_ok=True)


    print_legal(f"Projeto inicializado com sucesso em {project_dir}")
    print_legal(
        f"Agora atualize o arquivo {path.join(project_dir, 'config.yml')} com as configurações do projeto e do DeepLabCut",
        type="warn",
    )
