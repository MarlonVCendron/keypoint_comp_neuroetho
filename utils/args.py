import argparse
from os import path

parser = None
DEFAULT_PROJECT_SUBDIR = "projects"

def _append_project_subdir(p_dir_str: str | None) -> str | None:
    if p_dir_str is None:
        return None
    return path.join(DEFAULT_PROJECT_SUBDIR, p_dir_str)

def build_parser():
    global parser
    parser = argparse.ArgumentParser(description="Keypoint MoSeq")
    parser.add_argument(
        "--project-dir",
        help="Nome do diretório do projeto.",
        type=_append_project_subdir,
    )
    parser.add_argument("--model-name", type=str, default="teste", help="Nome base para o modelo.")
    parser.add_argument(
        "--mixed-map-iters",
        type=int,
        default=8,
        help="Número de batches. Diminui o uso de memória, mas aumenta o tempo de execução. Por padrão, 8.",
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Ação a ser executada. Use '<comando> -h' para mais ajuda sobre um comando específico.",
    )
    
    # Subparser para inicializar o projeto
    parser_init = subparsers.add_parser("init", help="Inicializa o projeto.")
    
    # Subparser para ajustar PCA
    parser_pca = subparsers.add_parser("fit_pca", help="Ajusta e salva PCA.")
    
    # Subparser para ajustar modelo AR
    parser_ar = subparsers.add_parser("fit_ar", help="Ajusta o modelo AR inicial.")
    parser_ar.add_argument(
        "--num-ar-iters",
        type=int,
        default=50,
        help="Número de iterações para o ajuste do modelo AR. Este também será o número da iteração do checkpoint.",
    )
    
    # Subparser para ajustar modelo AR-HMM
    parser_arhmm = subparsers.add_parser(
        "fit_arhmm", help="Ajusta o modelo AR-HMM, carregando um modelo AR pré-treinado."
    )
    parser_arhmm.add_argument(
        "--num-ar-iters-checkpoint",
        type=int,
        required=True,
        help="Número da iteração do checkpoint do modelo AR a ser carregado (ex: 50 se o modelo AR foi treinado por 50 iterações).",
    )
    parser_arhmm.add_argument(
        "--iters", type=int, default=500, help="Número de iterações para o ajuste do modelo AR-HMM."
    )
    parser_arhmm.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="Valor de Kappa para o modelo AR-HMM. Sobrescreve o valor no config.yml se fornecido.",
    )
    
    # Subparser para scan de kappa
    parser_kappa = subparsers.add_parser(
        "kappa_scan",
        help="Scan de kappa: ajusta AR-HMM para múltiplos valores de kappa.",
    )
    parser_kappa.add_argument(
        "--kappa-log-start",
        type=float,
        default=3,
        help="Valor inicial do expoente do kappa. Exemplo: 3 = 10^3 = 1000.",
    )
    parser_kappa.add_argument(
        "--kappa-log-end",
        type=float,
        default=7,
        help="Valor final do expoente do kappa. Exemplo: 7 = 10^7 = 10000000.",
    )
    parser_kappa.add_argument(
        "--num-kappas",
        type=int,
        default=5,
        help="Número de valores de kappa a serem testados. Exemplo: 5 = 1000, 10000, 100000, 1000000, 10000000.",
    )
    parser_kappa.add_argument(
        "--decrease-kappa-factor",
        type=float,
        default=10,
        help="Fator pelo qual o kappa é dividido após o ajuste do modelo AR. Por padrão, 10.",
    )
    parser_kappa.add_argument(
        "--num-ar-iters",
        type=int,
        default=50,
        help="Número de iterações para o ajuste do modelo AR.",
    )
    parser_kappa.add_argument(
        "--iters",
        type=int,
        default=250,
        help="Número de iterações para o ajuste do modelo AR-HMM.",
    )

    return parser

def get_args():
    return parser.parse_args()

def get_subparser(subparser_name):
    return parser.parse_args(subparser_name)

def get_arg(arg_name):
    return getattr(get_args(), arg_name, None)
