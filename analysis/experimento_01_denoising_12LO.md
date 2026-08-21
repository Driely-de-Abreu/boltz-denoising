# Experimento 01 — Inspeção do processo de denoising do Boltz2 com a proteína 12LO

## 1. Objetivo

Investigar o processo iterativo de denoising realizado pelo Boltz2 e observar o comportamento da variável `atom_coords_denoised` durante as etapas de sampling.

O objetivo inicial foi verificar se `atom_coords_denoised` é calculado em cada etapa do processo e investigar o formato das coordenadas produzidas pelo modelo.

Este experimento é uma etapa inicial para, posteriormente, estudar a possibilidade de armazenar as coordenadas intermediárias e construir uma trajetória do processo de denoising.

## 2. Proteína utilizada

- PDB ID: 12LO
- Arquivo de entrada: `inputs/teste_12LO.yaml`

## 3. Código investigado

O código investigado está localizado em:

`boltz-repo/src/boltz/model/modules/diffusionv2.py`

A investigação concentrou-se no loop de sampling da classe `AtomDiffusion`, especialmente na variável `atom_coords_denoised`.

## 4. Modificação realizada

Após a atribuição:

```python
atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk
```

foram adicionados temporariamente os seguintes prints:

```python
print("STEP:", step_idx)
print("DENOISED SHAPE:", atom_coords_denoised.shape)
```

Esses prints foram utilizados para observar:

- o índice da etapa atual (`step_idx`);
- o formato do tensor `atom_coords_denoised`.

## 5. Execução

Foi realizada uma execução da proteína 12LO utilizando o Boltz2.

- Modelo: Boltz2
- Versão do pacote: 2.2.1
- Proteína: 12LO
- Número de sampling steps: 200
- Número de amostras: 1
- Acelerador: CPU
- MSA: servidor ColabFold

A execução foi concluída com sucesso, sem exemplos com falha.

## 6. Resultado

Foram observados os steps:

```text
STEP: 0
...
STEP: 199
```

Em cada etapa foi observado:

```text
DENOISED SHAPE: torch.Size([1, 448, 3])
```

Portanto, nesta execução, `atom_coords_denoised` apresentou o formato `[1, 448, 3]` nas etapas observadas.

A última dimensão, de tamanho 3, está relacionada às três coordenadas espaciais utilizadas para representar as posições:

```text
[x, y, z]
```

O significado exato da dimensão de tamanho 448 ainda será investigado.

## 7. Observação

Neste primeiro experimento, as coordenadas intermediárias não foram armazenadas.

Os prints serviram apenas para verificar que `atom_coords_denoised` é calculado durante o processo iterativo e observar o formato do tensor produzido.

Portanto, ainda não temos uma trajetória armazenada das coordenadas.

## 8. Conclusão inicial

Foram observadas 200 etapas de sampling, de `step_idx = 0` até `step_idx = 199`.

Em cada etapa foi produzido um tensor `atom_coords_denoised` com formato:

```text
[1, 448, 3]
```

Esse resultado indica que `atom_coords_denoised` é uma variável importante para a investigação do processo de denoising e para o objetivo futuro de extrair as coordenadas intermediárias.

## 9. Próximos passos

- Investigar de onde vem o número de 200 sampling steps.
- Entender melhor o fluxo entre `atom_coords`, `atom_coords_noisy`, `atom_coords_denoised` e `atom_coords_next`.
- Investigar o significado das 448 posições.
- Determinar se `atom_coords_denoised` é a variável que devemos armazenar para construir a trajetória.
- Testar o armazenamento das coordenadas intermediárias.