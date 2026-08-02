#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho 2: Escolha do Modelo Probabilístico e Testes de Normalidade
Fases P1, P2, P3 e P4.
"""

import os
import sys
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors
from fitter import Fitter, get_common_distributions

def main():
    # Definindo configurações de exibição do pandas para tabelas bonitas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}' if abs(x) > 1e-4 else f'{x:.6e}')

    # Nome do arquivo de texto para salvar a saída do console
    log_filename = "resultados_trabalho2.txt"
    sys.stdout = Logger(log_filename)

    print("=" * 100)
    print(f"{'EXECUÇÃO DO TRABALHO 2: MODELOS PROBABILÍSTICOS E TESTES DE NORMALIDADE':^100}")
    print("=" * 100)

    # =============================================================================
    # P1 — ESCOLHA DO MODELO PROBABILÍSTICO (FITTER)
    # =============================================================================
    print("\n" + "#" * 100)
    print(f"{'Fase P1 — Ajuste e Seleção dos 3 Melhores Modelos Probabilísticos':^100}")
    print("#" * 100)

    # 1. Gerar dados sintéticos de uma distribuição Gamma
    np.random.seed(12012007)
    dados_reais = stats.gamma.rvs(a=2.5, scale=2.0, size=1000)
    
    print("\n[P1] Gerados 1000 dados a partir de uma distribuição teórica Gamma(a=2.5, scale=2.0).")
    print(f"Média Amostral: {np.mean(dados_reais):.4f}")
    print(f"Desvio Padrão Amostral: {np.std(dados_reais):.4f}\n")

    # 2. Inicializar o Fitter com distribuição comum
    print("Ajustando distribuições comuns utilizando Fitter...")
    f = Fitter(dados_reais, distributions=get_common_distributions(), timeout=30)
    f.fit()

    # 3. Construir a tabela consolidada de métricas
    linhas_tabela = []
    for nome_dist in f.fitted_param.keys():
        erro_quadratico = f.df_errors.loc[nome_dist, 'sumsquare_error']
        aic = f.df_errors.loc[nome_dist, 'aic']
        bic = f.df_errors.loc[nome_dist, 'bic']

        # Calcular teste Kolmogorov-Smirnov (KS) com parâmetros ajustados
        params = f.fitted_param[nome_dist]
        try:
            ks_stat, p_val = stats.kstest(dados_reais, nome_dist, args=params)
        except Exception:
            ks_stat, p_val = np.nan, np.nan

        linhas_tabela.append({
            'Distribuição': nome_dist,
            'Erro Quadrático': erro_quadratico,
            'AIC': aic,
            'BIC': bic,
            'KS Estatística': ks_stat,
            'KS Valor-p': p_val
        })

    df_final = pd.DataFrame(linhas_tabela)
    df_final.set_index('Distribuição', inplace=True)
    df_final = df_final.sort_values(by='BIC')

    print("\n=================================== TABELA DE MÉTRICAS ===================================")
    print(df_final)
    print("==========================================================================================")

    # 4. Identificar os 3 melhores modelos
    top_3 = df_final.index[:3].tolist()
    print(f"\n[P1] Os 3 melhores modelos probabilísticos (baseados no critério BIC) são:")
    for idx, dist in enumerate(top_3, 1):
        print(f"  {idx}. {dist} (BIC: {df_final.loc[dist, 'BIC']:.4f})")

    # 5. Salvar o gráfico do histograma ajustado
    plt.figure(figsize=(10, 6))
    f.summary(plot=True)
    plt.title("Ajuste de Modelos Probabilísticos (Fitter)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Valores de X", fontsize=11)
    plt.ylabel("Densidade de Probabilidade", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_p1_path = "histogram_fitter.png"
    plt.savefig(plot_p1_path, dpi=150)
    plt.close()
    print(f"\n[P1] Gráfico do histograma salvo com sucesso em '{plot_p1_path}'.")


    # =============================================================================
    # P2 & P3 — TESTES DE NORMALIDADE E Q-Q PLOT
    # =============================================================================
    print("\n" + "#" * 100)
    print(f"{'Fases P2 & P3 — Avaliação de Normalidade (Testes Estatísticos e Visual)':^100}")
    print("#" * 100)

    # Execução dos testes de normalidade
    alfa = 0.05
    
    # Shapiro-Wilk
    stat_sw, p_sw = stats.shapiro(dados_reais)
    decisao_sw = "Não Rejeita Normalidade" if p_sw >= alfa else "Rejeita Normalidade"

    # Kolmogorov-Smirnov (Lilliefors)
    stat_ll, p_ll = lilliefors(dados_reais, dist='norm')
    decisao_ll = "Não Rejeita Normalidade" if p_ll >= alfa else "Rejeita Normalidade"

    # D'Agostino-Pearson
    stat_dp, p_dp = stats.normaltest(dados_reais)
    decisao_dp = "Não Rejeita Normalidade" if p_dp >= alfa else "Rejeita Normalidade"

    # Anderson-Darling
    resultado_ad = stats.anderson(dados_reais, dist='norm')
    stat_ad = resultado_ad.statistic
    val_critico_5 = resultado_ad.critical_values[2]  # Índice 2 equivale a 5% de significância
    decisao_ad = "Não Rejeita Normalidade" if stat_ad <= val_critico_5 else "Rejeita Normalidade"

    print("\n" + "="*95)
    print(f"{'TABELA CONSOLIDADA DE TESTES DE NORMALIDADE (n=1000)':^95}")
    print("="*95)
    print(f"{'Nome do Teste Estatístico':<32} | {'Estatística':<12} | {'P-Valor / V. Crítico':<22} | {'Decisão Final (Alfa=5%)'}")
    print("-"*95)
    print(f"{'Shapiro-Wilk':<32} | {stat_sw:<12.4f} | {p_sw:<22.4e} | {decisao_sw}")
    print(f"{'Kolmogorov-Smirnov (Lilliefors)':<32} | {stat_ll:<12.4f} | {p_ll:<22.4e} | {decisao_ll}")
    print(f"{'D-Agostino-Pearson':<32} | {stat_dp:<12.4f} | {p_dp:<22.4e} | {decisao_dp}")
    print(f"{'Anderson-Darling':<32} | {stat_ad:<12.4f} | VC (5%): {val_critico_5:<14.3f} | {decisao_ad}")
    print("="*95 + "\n")
    
    print("[P2/P3] Interpretação dos testes:")
    print("  Como a amostra é grande (n=1000), os testes tradicionais de hipótese são extremamente sensíveis")
    print("  a desvios mínimos da normalidade, o que explica a forte rejeição (p-valores próximos a zero).")
    print("  Portanto, a análise gráfica pelo Q-Q Plot com bandas de confiança é fundamental.")

    # Criação do Q-Q plot com banda de confiança de 95%
    n = len(dados_reais)
    dados_ordenados = np.sort(dados_reais)
    posicoes_proporcionais = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    quantis_teoricos = stats.norm.ppf(posicoes_proporcionais)
    dados_padronizados = (dados_ordenados - np.mean(dados_reais)) / np.std(dados_reais)

    # Erro padrão dos quantis
    densidade_teorica = stats.norm.pdf(quantis_teoricos)
    erro_padrao = (1.0 / densidade_teorica) * np.sqrt((posicoes_proporcionais * (1.0 - posicoes_proporcionais)) / n)

    # Banda de confiança de 95% (Z = 1.96)
    z_critico = stats.norm.ppf(0.975)
    banda_superior = quantis_teoricos + z_critico * erro_padrao
    banda_inferior = quantis_teoricos - z_critico * erro_padrao

    plt.figure(figsize=(8, 6))
    plt.fill_between(quantis_teoricos, banda_inferior, banda_superior, color='gray', alpha=0.2, label='Banda de Confiança (95%)')
    plt.plot(quantis_teoricos, banda_superior, color='gray', linestyle='--', alpha=0.5)
    plt.plot(quantis_teoricos, banda_inferior, color='gray', linestyle='--', alpha=0.5)
    plt.plot(quantis_teoricos, quantis_teoricos, color='red', linewidth=2, label='Linha de Simetria Perfeita (Normalidade)')
    plt.scatter(quantis_teoricos, dados_padronizados, color='tab:blue', edgecolor='white', s=20, alpha=0.6, label='Dados Amostrais')

    plt.title("Q-Q Plot Avançado com Banda de Confiança (95%)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Quantis Teóricos (Distribuição Normal)", fontsize=11)
    plt.ylabel("Quantis Amostrais (Dados Padronizados)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    plot_p2_path = "qqplot_normalidade.png"
    plt.savefig(plot_p2_path, dpi=150)
    plt.close()
    print(f"[P2/P3] Gráfico Q-Q Plot salvo com sucesso em '{plot_p2_path}'.")


    # =============================================================================
    # P4 — CENÁRIOS (BAIXO, MÉDIO, ELEVADO) - ALTERNATIVA A
    # =============================================================================
    print("\n" + "#" * 100)
    print(f"{'Fase P4 — Definição de Cenários (Baixo, Médio, Elevado) - Alternativa A':^100}")
    print("#" * 100)

    # 1. Calcular quartis empíricos da amostra real
    Q1 = np.percentile(dados_reais, 25)
    Q2 = np.percentile(dados_reais, 50)  # Mediana
    Q3 = np.percentile(dados_reais, 75)

    print(f"\n[P4] Quartis empíricos calculados sobre os dados reais:")
    print(f"  Q1 (25%): {Q1:.4f}")
    print(f"  Q2 (50%): {Q2:.4f} (Mediana)")
    print(f"  Q3 (75%): {Q3:.4f}\n")

    # 2. Cenários Empíricos
    n_total = len(dados_reais)
    emp_baixo = np.sum(dados_reais <= Q1) / n_total
    emp_medio = np.sum((dados_reais > Q1) & (dados_reais < Q3)) / n_total
    emp_elevado = np.sum(dados_reais >= Q3) / n_total

    print("=" * 90)
    print(f"{'TABELA DE CENÁRIOS EMPÍRICOS (DADOS REAIS)':^90}")
    print("=" * 90)
    print(f"{'Cenário':<12} | {'Condição':<15} | {'Limite / Intervalo':<25} | {'Probabilidade Amostral'}")
    print("-" * 90)
    print(f"{'Baixo':<12} | {'X <= Q1':<15} | {f'<= {Q1:.4f}':<25} | {emp_baixo*100:.2f}%")
    print(f"{'Médio':<12} | {'Q1 < X < Q3':<15} | {f']{Q1:.4f}, {Q3:.4f}[':<25} | {emp_medio*100:.2f}%")
    print(f"{'Elevado':<12} | {'X >= Q3':<15} | {f'>= {Q3:.4f}':<25} | {emp_elevado*100:.2f}%")
    print("=" * 90 + "\n")

    # 3. Cenários Teóricos para as 3 melhores distribuições
    print("Calculando cenários teóricos comparativos (Alternativa A) para as 3 melhores distribuições:")
    
    comparacao_linhas = []
    
    for nome_dist in top_3:
        params = f.fitted_param[nome_dist]
        dist_obj = getattr(stats, nome_dist)
        
        # Calcular as probabilidades acumuladas teóricas nos limites Q1 e Q3
        # P(X <= Q1)
        prob_baixo_teorica = dist_obj.cdf(Q1, *params)
        # P(X <= Q3)
        prob_q3_teorica = dist_obj.cdf(Q3, *params)
        # P(Q1 < X < Q3)
        prob_medio_teorica = prob_q3_teorica - prob_baixo_teorica
        # P(X >= Q3)
        prob_elevado_teorica = 1.0 - prob_q3_teorica
        
        print("\n" + "-" * 80)
        print(f"Cenários Teóricos para o Modelo: {nome_dist.upper()}")
        print("-" * 80)
        print(f"{'Cenário':<12} | {'Fórmula Teórica':<20} | {'Probabilidade Estimada P(Cenário)'}")
        print("-" * 80)
        print(f"{'Baixo':<12} | {'P(X <= Q1)':<20} | {prob_baixo_teorica*100:.2f}%")
        print(f"{'Médio':<12} | {'P(Q1 < X < Q3)':<20} | {prob_medio_teorica*100:.2f}%")
        print(f"{'Elevado':<12} | {'P(X >= Q3)':<20} | {prob_elevado_teorica*100:.2f}%")
        
        # Guardar para tabela comparativa consolidada
        comparacao_linhas.append({
            'Modelo': nome_dist,
            'Baixo (%)': f"{prob_baixo_teorica*100:.2f}%",
            'Médio (%)': f"{prob_medio_teorica*100:.2f}%",
            'Elevado (%)': f"{prob_elevado_teorica*100:.2f}%"
        })
    print("-" * 80 + "\n")

    # 4. Tabela comparativa consolidada final
    df_comparativo = pd.DataFrame(comparacao_linhas)
    # Adicionar a linha empírica de referência
    df_comparativo.loc[len(df_comparativo)] = {
        'Modelo': 'Dados Amostrais (Real)',
        'Baixo (%)': f"{emp_baixo*100:.2f}%",
        'Médio (%)': f"{emp_medio*100:.2f}%",
        'Elevado (%)': f"{emp_elevado*100:.2f}%"
    }
    df_comparativo.set_index('Modelo', inplace=True)

    print("=" * 80)
    print(f"{'TABELA COMPARATIVA CONSOLIDADA DE CENÁRIOS':^80}")
    print("=" * 80)
    print(df_comparativo)
    print("=" * 80)
    print("\n[P4] Análise da Alternativa A:")
    print("  A tabela acima compara as probabilidades de cada cenário estimadas pelos modelos teóricos.")
    print("  Quanto mais próximo o modelo estimar as proporções dos Dados Amostrais (25% - 50% - 25%),")
    print("  melhor é a aderência prática do modelo probabilístico para modelar os cenários de risco.")
    
    print("\n" + "=" * 100)
    print(f"{'FIM DA EXECUÇÃO - DADOS SALVOS EM resultados_trabalho2.txt':^100}")
    print("=" * 100)

    # Restaura stdout original
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal


# Classe auxiliar para duplicar a saída do terminal (imprime na tela e salva no arquivo txt)
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == '__main__':
    main()
