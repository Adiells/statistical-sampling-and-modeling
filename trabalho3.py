#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho 3: Avaliação e Métricas de uma Matriz de Confusão com Intervalos de Confiança (95%)
Utilizando a biblioteca python 'confidenceinterval' (equivalente ao pacote 'epiR' do R).
"""

import numpy as np
import pandas as pd
from confidenceinterval import accuracy_score, recall_score, tnr_score, precision_score, f1_score

def calcular_metricas_com_biblioteca(tp, tn, fp, fn, conf_level=0.95):
    # 1. Reconstruir vetores y_true e y_pred correspondentes
    # Classe Negativa = 0 (Tumor Benigno), Classe Positiva = 1 (Tumor Maligno)
    y_true = np.array([0] * (tn + fp) + [1] * (tp + fn))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    
    # 2. Calcular Métricas e Intervalos de Confiança usando a biblioteca 'confidenceinterval'
    # Por padrão, usa-se o método Wald ('normal') para proporções simples e Bootstrapping para F1-Score
    acc, acc_ci = accuracy_score(y_true, y_pred, confidence_level=conf_level, method='normal')
    sens, sens_ci = recall_score(y_true, y_pred, confidence_level=conf_level, average='binary', method='normal')
    spec, spec_ci = tnr_score(y_true, y_pred, confidence_level=conf_level, method='normal')
    prec, prec_ci = precision_score(y_true, y_pred, confidence_level=conf_level, average='binary', method='normal')
    f1, f1_ci = f1_score(y_true, y_pred, confidence_level=conf_level, average='binary', method='bootstrap_bca')
    
    # 3. Montar o dataframe estruturado
    metricas = {
        'Métrica': ['Acurácia', 'Sensibilidade (Recall)', 'Especificidade', 'Precisão', 'F1-Score'],
        'Fórmula': [
            '(TP + TN) / Total', 
            'TP / (TP + FN)', 
            'TN / (TN + FP)', 
            'TP / (TP + FP)', 
            '2 * (Prec * Sens) / (Prec + Sens)'
        ],
        'Valor Estimado': [acc, sens, spec, prec, f1],
        '95% IC Inferior': [acc_ci[0], sens_ci[0], spec_ci[0], prec_ci[0], f1_ci[0]],
        '95% IC Superior': [acc_ci[1], sens_ci[1], spec_ci[1], prec_ci[1], f1_ci[1]]
    }
    
    df = pd.DataFrame(metricas)
    df['Valor Estimado (%)'] = (df['Valor Estimado'] * 100).map('{:.2f}%'.format)
    df['Intervalo de Confiança (95%)'] = df.apply(
        lambda r: f"[{r['95% IC Inferior'] * 100:.2f}%, {r['95% IC Superior'] * 100:.2f}%]", axis=1
    )
    return df

def main():
    tp, tn, fp, fn = 52, 91, 4, 3
    df_metricas = calcular_metricas_com_biblioteca(tp, tn, fp, fn)
    
    print("=" * 100)
    print(f"{'TRABALHO 3: MÉTRICAS DA MATRIZ DE CONFUSÃO E INTERVALOS DE CONFIANÇA (95%)':^100}")
    print(f"{'BIBLIOTECA UTILIZADA: confidenceinterval (EQUIVALENTE AO epiR DO R)':^100}")
    print("=" * 100)
    print(f"Valores de Entrada: TP={tp} (Maligno Correto), TN={tn} (Benigno Correto)")
    print(f"                    FP={fp} (Falso Alarme),  FN={fn} (Maligno Não Detectado)")
    print("-" * 100)
    
    df_print = df_metricas[['Métrica', 'Fórmula', 'Valor Estimado (%)', 'Intervalo de Confiança (95%)']]
    print(df_print.to_string(index=False))
    print("=" * 100)

if __name__ == '__main__':
    main()
