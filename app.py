import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard Logística", layout="wide")
st.title("📦 Decisão Logística: Análise de Risco e Trade-offs")
st.markdown("Simulador de Estoque para o Módulo X100 sob Incerteza de Demanda.")

# --- Parâmetros Fixos ---
mu, sigma, days, L = 50, 15, 365, 5
S, H_anual, C_f = 200.00, 2.00, 10.00
H_diario = H_anual / 365

np.random.seed(42)
demand = np.maximum(np.random.normal(mu, sigma, days), 0).round().astype(int)
D_total = demand.sum()
Q_otimo = round(np.sqrt((2 * D_total * S) / H_anual))
R_deterministico = mu * L

# --- Motor de Simulação ---
def simular_estoque(R_alvo):
    estoque_atual = Q_otimo
    pedidos_em_transito = []
    estoque_fisico = np.zeros(days)
    custo_pedido_total, custo_ruptura_total, unidades_perdidas = 0, 0, 0
    
    for t in range(days):
        for pedido in pedidos_em_transito[:]:
            if pedido['dia_chegada'] == t:
                estoque_atual += pedido['qtd']
                pedidos_em_transito.remove(pedido)
                
        demanda_dia = demand[t]
        if estoque_atual >= demanda_dia:
            estoque_atual -= demanda_dia
        else:
            falta = demanda_dia - estoque_atual
            unidades_perdidas += falta
            custo_ruptura_total += falta * C_f
            estoque_atual = 0 
            
        estoque_fisico[t] = estoque_atual
        posicao_estoque = estoque_atual + sum(p['qtd'] for p in pedidos_em_transito)
        if posicao_estoque <= R_alvo and len(pedidos_em_transito) == 0:
            pedidos_em_transito.append({'qtd': Q_otimo, 'dia_chegada': t + L})
            custo_pedido_total += S
            
    custo_manutencao_total = estoque_fisico.sum() * H_diario
    return estoque_fisico, custo_manutencao_total + custo_pedido_total + custo_ruptura_total, unidades_perdidas, custo_ruptura_total, custo_manutencao_total

# --- Interface Interativa (Barra Lateral) ---
st.sidebar.header("⚙️ Parâmetros de Decisão")
nivel_servico = st.sidebar.slider("Nível de Serviço Alvo (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1)

# --- Cálculos Dinâmicos ---
z_score = stats.norm.ppf(nivel_servico / 100)
SS = round(z_score * (sigma * np.sqrt(L)))
R_dinamico = R_deterministico + SS

# Roda a simulação para o cenário escolhido
estoque_sim, custo_total, faltas, custo_falta, custo_manutencao = simular_estoque(R_dinamico)

# --- Exibição dos KPIs ---
st.subheader("📊 Indicadores de Desempenho (Simulação 365 dias)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Estoque de Segurança (SS)", f"{SS} un.", "Proteção extra")
col2.metric("Ponto de Ressuprimento (R)", f"{R_dinamico} un.", f"Gatilho de compra")
col3.metric("Rupturas (Vendas Perdidas)", f"{faltas} un.", "Queda drástica no Custo de Falta!" if faltas == 0 else "Risco Operacional", delta_color="inverse")
col4.metric("Custo Total Logístico", f"R$ {custo_total:.2f}")

# --- Gráfico ---
st.subheader("📈 Dinâmica do Estoque Físico")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(estoque_sim, color='#2ca02c', linewidth=1.5, label='Nível de Estoque Simulado')
ax.axhline(0, color='red', linewidth=1, linestyle='--')
ax.set_ylabel("Unidades em Estoque")
ax.set_xlabel("Dias do Ano")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.success("💡 **Decisão Recomendada:** Observe como o aumento do Nível de Serviço zera as rupturas, equilibrando o *Trade-off* entre armazenagem e falta.")
