import argparse

# Pré-declarar todos os argumentos com None pra poder importá-los diretamente
project_dir = None
model_name = None
mixed_map_iters = None
command = None
num_ar_iters = None
num_ar_iters_checkpoint = None
iters = None
kappa = None
base_iters = None
kappa_values = None

parser = argparse.ArgumentParser(
    description="Keypoint MoSeq. Requer um diretório de projeto com arquivo de configuração do DeepLabCut (ex: `config.yml`) dentro dele."
)
parser.add_argument(
    "--project-dir",
    type=str,
    default="project",
    help="Diretório do projeto (deve conter config.yml).",
)
parser.add_argument("--model-name", type=str, default="teste", help="Nome base para o modelo.")
parser.add_argument(
    "--mixed-map-iters",
    type=int,
    default=8,
    help="Número de batches. Diminui o uso de memória em X vezes, aumenta o tempo de execução em X vezes",
)

subparsers = parser.add_subparsers(
    dest="command",
    required=True,
    help="Ação a ser executada. Use '<comando> -h' para mais ajuda sobre um comando específico.",
)

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

# Subparser para varredura de kappa
parser_kappa = subparsers.add_parser(
    "kappa_scan",
    help="Realiza uma varredura de kappa: ajusta AR-HMM para múltiplos valores de kappa.",
)
parser_kappa.add_argument(
    "--num-ar-iters-checkpoint",
    type=int,
    required=True,
    help="Número da iteração do checkpoint do modelo AR a ser carregado para cada varredura.",
)
parser_kappa.add_argument(
    "--base-iters",
    type=int,
    default=200,
    help="Número de iterações para cada modelo AR-HMM na varredura.",
)
parser_kappa.add_argument(
    "--kappa-values",
    type=str,
    required=True,
    help="Lista de valores de kappa separados por vírgula para varredura (ex: '1e3,1e4,1e5').",
)


parsed_cli_args = parser.parse_args()

# Expõe os argumentos como variáveis globais para serem importadas diretamente
for arg_name, arg_value in vars(parsed_cli_args).items():
    globals()[arg_name] = arg_value
