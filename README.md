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

##### Ajustar PCA

##### Scan do kappa

##### Treinamento do AR

##### Treinamento do AR-HMM









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
