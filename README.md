# Triagem de materiais

Aplicação em Python para identificar o material predominante em uma foto de resíduo. Aceita papel (incluindo papelão), plástico, metal, vidro e orgânico. Envie uma foto ou escolha uma das três amostras para experimentar.

## Executar

Na pasta do repositório, com Python 3.13:

```sh
python -m venv .venv
```

Ative o ambiente com `.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate` no Linux/macOS. Em seguida:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abra o endereço exibido no terminal. Na primeira classificação, a aplicação baixa cerca de 95 MB de pesos (89 MB do filtro visual e 5,8 MB do classificador) do Hugging Face. A revisão fica armazenada no cache local e pode ser reutilizada offline. A foto é processada em memória na máquina que executa o Streamlit.

## Como a classificação funciona

`app.py` cuida do envio, das amostras e da apresentação. `classificador.py` carrega a rede pré-treinada [WasteWise YOLOv8n-cls](https://huggingface.co/SriramRokkam/wastewise-garbage-cls/tree/a30c36c6b181ac0d2eb387bbd4f6d4a5b88ee078) e executa a inferência em CPU com ONNX Runtime. A folha de estilos está em `interface.css`. A capa ilustrativa em `static/materiais.webp` foi gerada para o projeto. A aplicação corrige a orientação da foto, converte para RGB e envia à rede um tensor de 224 × 224 pixels com valores entre 0 e 1.

Antes da classificação, o filtro [CLIP ViT-B/32](https://huggingface.co/Xenova/clip-vit-base-patch32/tree/6b7c523703ae26e39be4b03774966f8571c7f523) compara a cena inteira com descrições de resíduos e de outros assuntos, incluindo retratos, animais e paisagens. Se a similaridade com resíduos for baixa ou ambígua, a tela mostra **Resíduo não identificado** sem atribuir material ou percentual. O filtro usa limiares heurísticos e pode aceitar imagens fora do domínio ou rejeitar resíduos legítimos.

Para imagens aceitas, a rede de materiais produz oito escores. A aplicação soma papel e papelão em **Papel**, e bateria e rejeitos em **Fora das categorias**. As demais saídas correspondem a um material cada. Se o maior escore agrupado ficar abaixo de 65%, a tela mostra **Identificação inconclusiva**. Esse limiar é uma escolha de interface e ainda precisa de calibração para uso operacional.

Use uma foto de um objeto por vez. A rede classifica a imagem inteira e pode confundir materiais semelhantes, mesmo com escore alto. Uma imagem sem resíduos também pode receber um rótulo. O escore expressa a preferência da rede naquela foto.

## Amostras

As três fotografias de demonstração vêm do [TrashNet](https://github.com/garythung/trashnet), arquivos `metal1.jpg`, `paper1.jpg` e `glass1.jpg`. A [licença das imagens](amostras/LICENSE) acompanha as cópias. As amostras passam pela mesma rede usada para as fotos enviadas.

## Verificação

Execute `python -m unittest discover -s tests` para verificar a rejeição de um retrato público e a classificação das três amostras. A foto de teste é uma versão reduzida de [astronaut.png, da NASA, distribuída pelo scikit-image](https://github.com/scikit-image/scikit-image/blob/v0.23.2/skimage/data/astronaut.png).

Os vetores de texto estão em `referencias_clip.npz`. Para regenerá-los após editar as descrições, instale `tokenizers==0.23.3` e rode `python gerar_referencias.py`. Esse passo baixa o codificador de texto e só é necessário para atualizar as referências.
