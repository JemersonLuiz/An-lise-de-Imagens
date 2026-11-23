# Análise Técnica e Oportunidades de Melhoria — `An-lise-de-Imagens`

Análise realizada nos arquivos públicos da branch `Análise-de-imagens` (`app.py`, `classificador.py`, `gerar_referencias.py`, `interface.css`, `requirements.txt`, `.streamlit/config.toml` e `tests/test_triagem.py`).

## 1. Prioridade Alta

### 1.1. Cobertura de testes unitários offline
- **Arquivo:** `tests/test_triagem.py`
- **Problema:** A suíte atual possui apenas 2 testes de integração que dependem de conexão com a internet para baixar ~95 MB de modelos do Hugging Face em `setUpClass`. Fluxos críticos não são testados isoladamente.
- **Melhoria sugerida:** Separar testes unitários rápidos (com mocks para `ort.InferenceSession`) cobrindo:
  - `ler_foto`: rejeição acima de 24 megapixels, conversão de PNG/WebP com transparência (RGBA) sobre fundo branco, rotação EXIF e erro em arquivos corrompidos;
  - `classificar`: soma correta de `Papel` + `Papelão`, tratamento de classes `None` (`Fora das categorias`), limiar `LIMIAR = 0.65` e rejeição pelo filtro `parece_residuo`.

### 1.2. Pré-processamento da imagem no classificador de materiais
- **Arquivo:** `classificador.py` (linha 79)
- **Problema:** Enquanto `parece_residuo()` usa `ImageOps.pad` (preservando proporção), `classificar()` usa `imagem.resize((224, 224), Image.Resampling.BILINEAR)`, o que distorce fotos retangulares (ex.: 4:3 ou 16:9 de celular) e pode degradar a precisão em objetos alongados (garrafas, latas).
- **Melhoria sugerida:** Avaliar pré-processamento que preserve proporção (center-crop ou letterbox compatível com o treino do `wastewise-garbage-cls`) e validar o impacto em um conjunto rotulado de imagens.

### 1.3. Recuperação de erro e cache de inferência no Streamlit
- **Arquivo:** `app.py` (linhas 86–93)
- **Problema:**
  - Ao falhar o carregamento da rede, o botão `"Tentar novamente"` executa `st.rerun()` sem limpar explicitamente `rede_em_cache.clear()`.
  - A inferência (`classificar`) roda novamente em qualquer rerun da interface para a mesma imagem, pois o resultado não é cacheado por hash da fotografia.
- **Melhoria sugerida:** Chamar `rede_em_cache.clear()` no fluxo de retentativa e cachear o resultado da classificação por hash dos bytes da imagem (ou em `st.session_state`).

---

## 2. Prioridade Média

### 2.1. Proteção numérica e explicabilidade do resultado
- **Arquivos:** `classificador.py` (linhas 69 e 83–87), `gerar_referencias.py` (linha 49), `app.py` (linhas 104–107)
- **Problema:**
  - `vetor / np.linalg.norm(vetor)` não protege contra norma zero em entradas anômalas.
  - A função `classificar()` retorna apenas o melhor rótulo; quando o resultado é inconclusivo (`escore < LIMIAR`), o usuário não vê quais materiais disputaram a classificação.
- **Melhoria sugerida:** Proteger a normalização contra divisão por zero e retornar a distribuição completa (ou top-3 materiais) para enriquecer o painel de resultado, além de orientações rápidas de descarte conforme a coleta seletiva.

### 2.2. Calibração reproduzível e configuração externa de limiares
- **Arquivo:** `classificador.py` (linhas 17–19 e 33)
- **Problema:** `LIMIAR`, `SIMILARIDADE_MINIMA`, `MARGEM_MINIMA` e `intra_op_num_threads = 2` estão fixos no código-fonte.
- **Melhoria sugerida:** Permitir sobrescrita via variáveis de ambiente (com defaults atuais) e incluir um script de avaliação (`avaliar.py`) que gere matriz de confusão e métricas (precisão/recall/F1) para calibrar os limiares com dados reais.

### 2.3. Completar galeria de amostras
- **Arquivo:** `app.py` (linhas 13–17) e pasta `amostras/`
- **Problema:** A aplicação classifica 5 materiais (Papel, Plástico, Metal, Vidro e Orgânico), mas a galeria de demonstração traz apenas 3 exemplos (`metal.jpg`, `papel.jpg`, `vidro.jpg`).
- **Melhoria sugerida:** Incluir amostras com licença compatível para **Plástico** e **Orgânico**, cobrindo todas as categorias suportadas.

---

## 3. Prioridade Baixa / Engenharia e DevOps

- **CI/CD no GitHub Actions:** Configurar workflow para rodar lint (`ruff`), checagem de tipos (`mypy`) e testes unitários offline a cada push/PR.
- **Padronização da branch padrão:** A branch padrão `Análise-de-imagens` contém acento e letra maiúscula, o que ocasionalmente causa problemas em scripts de CI e terminais. Padronizar para `main` reduz atritos.
- **Separação de dependências e containerização:** Criar `requirements-dev.txt` (incluindo `tokenizers`, `ruff`, `pytest`) separado do `requirements.txt` de produção, além de um `Dockerfile` para facilitar deploy reproduzível.
- **Fragilidade de seletores CSS internos (`interface.css`):** O arquivo depende de seletores internos do Streamlit (`[data-testid="stMainBlockContainer"]`, `[data-testid="stFileUploaderDropzone"]`, etc.). Documentar essa dependência ao atualizar a versão do `streamlit` evita quebras visuais inesperadas.
