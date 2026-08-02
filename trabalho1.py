#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho 1: Técnicas de Amostragem sobre a População
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 100)
    print(f"{'TRABALHO 1: AVALIAÇÃO DE TÉCNICAS DE AMOSTRAGEM':^100}")
    print("=" * 100)

    # 1. Criação de uma população fictícia para demonstração
    np.random.seed(12012007)
    tamanho_populacao = 1000
    dados_pop = {
        'ID': range(1, tamanho_populacao + 1),
        'Valor': np.random.normal(loc=100, scale=15, size=tamanho_populacao),
        'Estrato': np.random.choice(['Grupo A', 'Grupo B', 'Grupo C'], size=tamanho_populacao, p=[0.5, 0.3, 0.2])
    }
    populacao = pd.DataFrame(dados_pop)
    tamanho_amostra = 100

    print(f"\n[População] Gerada com sucesso: {tamanho_populacao} registros.")
    print(f"Média Populacional Real: {populacao['Valor'].mean():.4f}")
    print(f"Variância Populacional Real: {populacao['Valor'].var():.4f}\n")

    # --- AMOSTRAGEM ALEATÓRIA SIMPLES (AAS) ---
    amostra_simples = populacao.sample(n=tamanho_amostra, random_state=12012007)
    print(f"1. Amostra Aleatória Simples: Selecionados {len(amostra_simples)} elementos.")
    print(f"   Média Amostral: {amostra_simples['Valor'].mean():.4f} | Variância Amostral: {amostra_simples['Valor'].var():.4f}\n")

    # --- AMOSTRAGEM SISTEMÁTICA (AS) ---
    k = tamanho_populacao // tamanho_amostra
    # Determinar início aleatório entre 0 e k-1 com semente fixada
    np.random.seed(12012007)
    inicio = np.random.randint(0, k)
    indices_sistematicos = np.arange(inicio, tamanho_populacao, step=k)[:tamanho_amostra]
    amostra_sistematica = populacao.iloc[indices_sistematicos]
    print(f"2. Amostra Sistemática (Passo k={k}, Início={inicio}): Selecionados {len(amostra_sistematica)} elementos.")
    print(f"   Média Amostral: {amostra_sistematica['Valor'].mean():.4f} | Variância Amostral: {amostra_sistematica['Valor'].var():.4f}\n")

    # --- AMOSTRAGEM ESTRATIFICADA (AE) ---
    proporcoes = populacao['Estrato'].value_counts(normalize=True)
    amostra_estratificada = pd.DataFrame()

    for estrato, prop in proporcoes.items():
        n_estrato = int(np.round(prop * tamanho_amostra))
        sub_pop = populacao[populacao['Estrato'] == estrato]
        amostra_sub = sub_pop.sample(n=n_estrato, random_state=12012007)
        amostra_estratificada = pd.concat([amostra_estratificada, amostra_sub])

    print(f"3. Amostra Estratificada Proporcional: Selecionados {len(amostra_estratificada)} elementos.")
    print(f"   Média Amostral: {amostra_estratificada['Valor'].mean():.4f} | Variância Amostral: {amostra_estratificada['Valor'].var():.4f}")
    print("\n   Distribuição de frequências na amostra:")
    for estrato, count in amostra_estratificada['Estrato'].value_counts().items():
        print(f"     - {estrato}: {count} elementos ({count/len(amostra_estratificada)*100:.2f}%)")
    
    print("-" * 100)
    print("\nTabela Resumo de Métricas de Amostragem:")
    tabela_dados = {
        'Técnica': ['AAS', 'Sistemática', 'Estratificada'],
        'Tamanho Amostra': [len(amostra_simples), len(amostra_sistematica), len(amostra_estratificada)],
        'Média Amostral': [amostra_simples['Valor'].mean(), amostra_sistematica['Valor'].mean(), amostra_estratificada['Valor'].mean()],
        'Variância Amostral': [amostra_simples['Valor'].var(), amostra_sistematica['Valor'].var(), amostra_estratificada['Valor'].var()],
        'Erro Absoluto (Média)': [
            abs(amostra_simples['Valor'].mean() - populacao['Valor'].mean()),
            abs(amostra_sistematica['Valor'].mean() - populacao['Valor'].mean()),
            abs(amostra_estratificada['Valor'].mean() - populacao['Valor'].mean())
        ]
    }
    df_resumo = pd.DataFrame(tabela_dados)
    print(df_resumo.to_string(index=False))
    print("=" * 100)

if __name__ == '__main__':
    main()
