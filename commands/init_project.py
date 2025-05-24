import keypoint_moseq as kpms
from utils.print_legal import print_legal
from utils.args import parser_init
from os import path


def init_project(project_dir_path):
    """Inicializa o projeto."""

    if not project_dir_path:
        print_legal("Diretório do projeto não informado (--project-dir)", type="error")
        parser_init.print_help()
        return

    kpms.setup_project(project_dir_path)

    print_legal(f"Projeto inicializado com sucesso em {project_dir_path}")
    print_legal(
        f"Agora atualize o arquivo {path.join(project_dir_path, 'config.yml')} com as configurações do projeto e do DeepLabCut",
        type="warn",
    )
