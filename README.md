# Statistical Sampling and Modeling

---

## 📂 Estrutura do Repositório

O repositório está organizado nos seguintes arquivos:

### 1. Relatório
- **[RELATORIO_UNIFICADO.md](RELATORIO_UNIFICADO.md):** Relatório consolidado em formato Markdown contendo as respostas teóricas, tabelas de dados de amostragem, modelagem probabilística, testes de normalidade, análise crítica de métricas e os códigos correspondentes.

### 2. Scripts Executáveis (Python)
- **[trabalho1.py](trabalho1.py):** Implementação de técnicas de amostragem sobre populações de dados:
  - Amostragem Aleatória Simples (AAS) ($n=100$)
  - Amostragem Sistemática ($n=100$)
  - Amostragem Estratificada Proporcional ($n=101$)
- **[trabalho2.py](trabalho2.py):** Ajuste de modelos probabilísticos e testes de hipóteses de normalidade:
  - Ajuste de 10 distribuições comuns usando a biblioteca `fitter`.
  - Testes de hipóteses de normalidade: Shapiro-Wilk, Kolmogorov-Smirnov (Lilliefors), D'Agostino-Pearson e Anderson-Darling.
  - Exportação de gráfico de histograma (`histogram_fitter.png`) e Q-Q Plot com banda de confiança de 95% (`qqplot_normalidade.png`).
  - Geração de cenários de risco/desempenho (Baixo, Médio, Elevado) baseados em quartis empíricos e teóricos (via CDF teórica — Alternativa A).
- **[trabalho3.py](trabalho3.py):** Avaliação de classificadores e inferência estatística de intervalos de confiança:
  - Cálculo de métricas da Matriz de Confusão: Acurácia, Sensibilidade (Recall), Especificidade, Precisão e F1-Score.
  - Intervalos de confiança de 95% analíticos (Wald/Normal) para proporções simples usando `statsmodels`.
  - Intervalo de confiança de 95% empírico (Bootstrapping com 5000 reamostragens) para F1-Score utilizando a biblioteca `confidenceinterval`.

### 3. Resultados Auxiliares e Gráficos
- **[resultados_trabalho2.txt](resultados_trabalho2.txt):** Logs textuais da execução das análises probabilísticas e tabelas de cenários do Trabalho 2.
- **[histogram_fitter.png](histogram_fitter.png):** Gráfico do ajuste das distribuições probabilísticas sobre os dados.
- **[qqplot_normalidade.png](qqplot_normalidade.png):** Gráfico Q-Q Plot avançado com bandas de confiança.

---

## 🛠️ Como Executar os Scripts

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado. O arquivo [requirements.txt](requirements.txt) lista todas as bibliotecas necessárias. 

Você pode criar um ambiente virtual e instalá-las usando:

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate  # No Linux/macOS
# .venv\Scripts\activate   # No Windows

# Instalar dependências
pip install -r requirements.txt
```

### Execução dos scripts
Para rodar cada etapa do trabalho e exibir os outputs no terminal, execute:

```bash
python trabalho1.py
python trabalho2.py
python trabalho3.py
```
