# Neuroetologia Computacional

:)

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
