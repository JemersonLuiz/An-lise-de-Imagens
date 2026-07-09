# Classificador de Residuos com IA

## Descricao

Projeto educacional de Visao Computacional para classificar imagens de residuos reciclaveis usando aprendizado por transferencia com MobileNetV2.

O usuario envia uma imagem e o sistema retorna a categoria prevista, a confianca do modelo e a probabilidade de cada classe.

## Objetivo

Criar uma aplicacao completa de Machine Learning capaz de:

- organizar imagens por categorias;
- treinar um modelo de classificacao de imagens;
- salvar o modelo treinado;
- prever novas imagens;
- exibir uma interface web simples com Streamlit.

## Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- Pillow
- NumPy
- Matplotlib
- Scikit-learn

## Estrutura do Projeto

```text
classificador-residuos-ia/
|-- data/
|   |-- train/
|   |   |-- papel/
|   |   |-- plastico/
|   |   `-- metal/
|   `-- validation/
|       |-- papel/
|       |-- plastico/
|       `-- metal/
|-- models/
|-- src/
|   |-- train.py
|   |-- predict.py
|   |-- evaluate.py
|   `-- utils.py
|-- app.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Como Organizar o Dataset

Coloque as imagens manualmente nas pastas abaixo antes de treinar:

```text
data/train/papel/
data/train/plastico/
data/train/metal/

data/validation/papel/
data/validation/plastico/
data/validation/metal/
```

Para adicionar novas classes no futuro, crie a mesma pasta dentro de `data/train/` e `data/validation/`.

## Como Instalar

```bash
git clone https://github.com/seu-usuario/classificador-residuos-ia.git
cd classificador-residuos-ia
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

No Linux ou Mac, ative o ambiente com:

```bash
source venv/bin/activate
```

## Como Treinar o Modelo

Antes de treinar, coloque as imagens dentro da pasta `data/`.

Depois execute:

```bash
python src/train.py
```

Ao final, o modelo sera salvo em:

```text
models/modelo_residuos.keras
```

Os nomes das classes serao salvos em:

```text
models/classes.txt
```

## Como Executar a Aplicacao

```bash
streamlit run app.py
```

Depois abra o endereco exibido pelo Streamlit no navegador.

## Exemplo de Uso

Entrada:

```text
Imagem de uma garrafa plastica
```

Saida esperada:

```text
Classe prevista: plastico
Confianca: 92.41%

Probabilidades:
papel: 3.20%
plastico: 92.41%
metal: 4.39%
```

## Avaliacao do Modelo

Para avaliar o modelo treinado:

```bash
python src/evaluate.py
```

O script mostra acuracia, precision, recall, f1-score e salva a matriz de confusao em:

```text
models/matriz_confusao.png
```

## Possiveis Melhorias

- Aumentar a quantidade de imagens.
- Adicionar novas categorias, como vidro e organico.
- Melhorar o tratamento de imagens de baixa qualidade.
- Criar uma API para integracao com outros sistemas.
- Fazer deploy da aplicacao na nuvem.
- Usar camera em tempo real.

## Aviso

Este projeto tem finalidade educacional e experimental. Ele nao deve ser usado como sistema profissional de reciclagem sem uma base de dados maior, testes mais rigorosos e validacao em ambiente real.

## Autor

Desenvolvido por Jemerson Luiz.
