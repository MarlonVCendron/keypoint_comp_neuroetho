# Neuroetologia Computacional 🐀

Implementação do [Keypoint MoSeq](https://github.com/dattalab/keypoint-moseq) pelo
[LEMCOG](https://sites.google.com/academico.ufpb.br/lemcog).

## Ajuda

Para mais detalhes sobre os comandos disponíveis, execute `python -m main -h`.

Para mais detalhes sobre a configuração do projeto, consulte a [documentação do Keypoint MoSeq](https://keypoint-moseq.readthedocs.io).

## Treinando um modelo

Passos para treinar um modelo e gerar os resultados:

##### Ambiente conda

Inicie criando um novo ambiente conda a partir do arquivo `env.yml` e ativando-o:

```sh
conda env create -f env.yml
conda activate keypoint_comp_neuroetho
```

##### Inicializar o projeto

O próximo passo é inicializar o projeto. Para isso, basta executar o principal comando `python -m main` passando o argumento `init` e o diretório do projeto como argumento.

```sh
python -m main --project-dir <nome_do_projeto> init
```

Isso criará o diretório `projects/<nome_do_projeto>` com o arquivo de configuração `config.yml` e o diretório `data`. Coloque os
vídeos e dados do DeepLabCut no diretório `data` e atualize o
arquivo `config.yml` com as configurações necessárias, principalmente:
- `bodyparts`, `use_bodyparts`, `skeleton`, `anterior_bodyparts`, `posterior-bodyparts` conforme os pontos usados no DeepLabCut.
- `video_dir`: o/os diretório/diretórios onde estão os vídeos e dados do DeepLabCut.
- `fps` dos vídeos.

##### Calibração de ruído

É necessário calibrar o ruído dos dados do DeepLabCut através do notebook `noise_calibration.ipynb`.

##### Ajustar PCA

Em seguida, ajuste o PCA com o comando:

```sh
python -m main --project-dir <nome_do_projeto> fit_pca
```

É preciso ajustar o PCA somente uma vez por projeto, o resultado é salvo no diretório `projects/<nome_do_projeto>/pca.p`. O
comando também irá printar quanto da variância é explicada pelo número de PCs; recomenda-se usar o número de PCs que explica 90%
da variância.

Ajuste em `projects/<nome_do_projeto>/config.yml` o parâmetro `latent_dim` com o número de PCs que você escolheu.


##### Scan do kappa

O hiperparâmetro `kappa` controla a frequência de transições entre os estados do modelo. Um valor de `kappa` maior resultará em um
modelo com menos transições e sílabas mais longas. Use o comando `kappa_scan` para treinar vários modelos com diferentes valores
de `kappa` para ajudar a encontrar o valor ideal:

```sh
python -m main --project-dir <nome_do_projeto> kappa_scan \
    --kappa-log-start 3 \
    --kappa-log-end 7 \
    --num-kappas 5 \
    --num-ar-iters 50 \
    --iters 200
```

Este comando irá treinar 5 modelos com `kappa` variando de $10^3$ a $10^7$. Os parâmetros `--num-ar-iters` e `--num-full-iters` controlam o número de iterações para o ajuste dos modelos AR e AR-HMM, respectivamente.

##### Treinamento do modelo completo

Treina o modelo completo, juntando [AR-HMM](treinamento-do-arhmm) e [modelo Keypoint MoSeq](treinamento-do-modelo-keypoint-moseq).

```sh
python -m main --project-dir <nome_do_projeto> --model-name <nome_do_modelo> fit_full_model \
  --ar-iters <número_de_iterações_do_ar> \
  --ar-kappa <valor_de_kappa_do_ar> \
  --iters <número_de_iterações_do_keypoint> \
  --kappa <valor_de_kappa_do_keypoint>
```


###### Treinamento do AR-HMM

É recomendado inicialmente treinar apenas o modelo AR-HMM por algumas iterações utilizando o comando a seguir e o parâmetro
`--iters` para definir o número de iterações. Se não especificado, o valor padrão é 50. Você também pode especificar
o nome do modelo com o parâmetro `--model-name`.

```sh
python -m main --project-dir <nome_do_projeto> --model-name <nome_do_modelo> fit_arhmm --num-ar-iters <número_de_iterações>
```

###### Treinamento do modelo Keypoint MoSeq

Treina o modelo Keypoint MoSeq, carregando o modelo AR-HMM pré-treinado.

```sh
python -m main --project-dir <nome_do_projeto> --model-name <nome_do_modelo> fit_keypoint --checkpoint <número_da_iteração_do_checkpoint_do_modelo> --iters <número_de_iterações> --kappa <valor_de_kappa>
```

##### Gerar os resultados

Gera os resultados do modelo, incluindo as animações e os gráficos.

```sh
python -m main --project-dir <nome_do_projeto> --model-name <nome_do_modelo> results --checkpoint <número_da_iteração_do_checkpoint_do_modelo>
```

##### Validação

Validação com Simba e dados de rearing.

```sh
python -m main --project-dir <nome_do_projeto> --model-name <nome_do_modelo> validation --checkpoint <número_da_iteração_do_checkpoint_do_modelo>
```

---

#### Conda

##### Criar o ambiente

```sh
conda env create -f env.yml
```


##### Ativar o ambiente

```sh
conda activate keypoint_comp_neuroetho
```


##### Atualizar o ambiente

```sh
conda env update -f env.yml
```

##### Exportar o ambiente

```sh
conda env export --no-builds > env.yml
```

---

##### Limite de memória

É possível limitar quanta memória da GPU o JAX pode usar.

```sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

##### Erros comuns

- `AttributeError: jax.tree_map was removed in JAX v0.6.0: use jax.tree.map (jax v0.4.25 or newer) or jax.tree_util.tree_map (any
  JAX version).`

Solução fuleira: Substituir ocorrências de `jax.tree_map` por `jax.tree.map` em `.../envs/keypoint_comp_neuroetho/lib/python3.10/site-packages/jax_moseq/utils/debugging.py`