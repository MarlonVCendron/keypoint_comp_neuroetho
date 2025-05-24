# Neuroetologia Computacional 🐀

## Treinando um modelo

Para treinar um modelo, é preciso trazer os vídeos e os dados processados do DeepLabCut para o diretório do projeto:
1. Coloque os dados dentro do diretório `DLC/`
2. basta criar um novo diretório com o nome do projeto dentro de `projects/`.








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



##### Limite de memória

É possível limitar quanta memória da GPU o JAX pode usar.

```sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```
