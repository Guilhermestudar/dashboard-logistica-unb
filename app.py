import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Logística X100 - UnB", layout="wide")
st.title("📦 Dashboard Avançado: Análise de Risco e Trade-offs Logísticos")
st.markdown("Simulação Estocástica de Estoques - Módulo X100 | LogiTech Distribuidora")

# --- MENU LATERAL (FILTROS) ---
st.sidebar.header("⚙️ Parâmetros de Simulação")
st.sidebar.markdown("Ajuste as variáveis de mercado e operação:")

nivel_servico_alvo = st.sidebar.slider("Meta de Nível de Serviço (%)", 50.0, 99.9, 95.0, 0.1)
sigma = st.sidebar.slider("Volatilidade da Demanda (Desvio Padrão)", 5, 30, 15, 1)
L = st.sidebar.slider("Lead Time do Fornecedor (Dias)", 1, 15, 5, 1)
C_f = st.sidebar.number_input("Custo de Ruptura (R$/un. perdida)", value=10.0, step=1.0)

# Parâmetros Fixos Base
mu = 50
days = 365
S = 200.00
H_anual = 2.00
H_diario = H_anual / 365

# Geração de Demanda (Dinâmica baseada no sigma escolhido)
np.random.seed(42)
demand = np.maximum(np.random.normal(mu, sigma, days), 0).round().astype(int)
D_total = demand.sum()

# Cálculos Teóricos
Q_otimo = round(np.sqrt((2 * D_total * S) / H_anual))
R_deterministico = mu * L

# Cálculo do Estoque de Segurança
z_score = stats.norm.ppf(nivel_servico_alvo / 100)
SS = round(z_score * (sigma * np.sqrt(L)))
R_estocastico = R_deterministico + SS

# --- MOTOR DE SIMULAÇÃO ---
def simular(R_alvo):
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
            
    custo_manutencao = estoque_fisico.sum() * H_diario
    custo_total = custo_manutencao + custo_pedido_total + custo_ruptura_total
    ns_real = 100 * (1 - (unidades_perdidas / D_total))
    return estoque_fisico, custo_total, unidades_perdidas, custo_ruptura_total, custo_manutencao, ns_real

# Rodar Cenários A (Sem proteção) e B (Com proteção)
estoque_A, custo_A, faltas_A, cr_A, cm_A, ns_A = simular(R_deterministico)
estoque_B, custo_B, faltas_B, cr_B, cm_B, ns_B = simular(R_estocastico)

# --- VISUALIZAÇÃO DOS RESULTADOS ---
st.subheader("📊 Comparativo de Desempenho (Horizonte de 365 Dias)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Estoque de Segurança Gerado", f"{SS} un.")
col2.metric("Nível de Serviço Real", f"{ns_B:.2f}%", f"{ns_B - ns_A:.2f}% vs Cenário A")
col3.metric("Vendas Salvas (Ruptura Evitada)", f"{faltas_A - faltas_B} un.", "Melhoria Operacional")
col4.metric("Economia Gerada (Custo Total)", f"R$ {custo_A - custo_B:.2f}", "Redução de Custo", delta_color="inverse")

st.divider()

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**📈 Dinâmica do Estoque: Cenário A vs Cenário B**")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(estoque_A, color='#d62728', alpha=0.5, linestyle='--', label='A: Determinístico (Rupturas)')
    ax1.plot(estoque_B, color='#2ca02c', linewidth=2, label='B: Estocástico (Com Proteção)')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylabel("Unidades Físicas")
    ax1.set_xlabel("Dias do Ano")
    ax1.legend(loc="upper right", fontsize='small')
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

with col_chart2:
    st.markdown("**💰 Composição do Custo Total Logístico**")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    categorias = ['Cenário A (Teórico)', 'Cenário B (Recomendado)']
    manutencao = [cm_A, cm_B]
    ruptura = [cr_A, cr_B]
    
    ax2.bar(categorias, manutencao, label='Custo de Armazenagem', color='#1f77b4')
    ax2.bar(categorias, ruptura, bottom=manutencao, label='Custo de Ruptura (Falta)', color='#ff7f0e')
    ax2.set_ylabel("Valor (R$)")
    ax2.legend(loc="upper right", fontsize='small')
    st.pyplot(fig2)

st.divider()

st.markdown("**🎯 Curva de Trade-off: Nível de Serviço vs Custo Total**")
st.markdown("*A simulação executa 20 cenários diferentes em background para plotar a curva de risco logístico.*")

ns_testes = np.linspace(80, 99.9, 20)
custos_testes = []
for ns_t in ns_testes:
    z_t = stats.norm.ppf(ns_t / 100)
    ss_t = round(z_t * (sigma * np.sqrt(L)))
    r_t = R_deterministico + ss_t
    _, c_tot, _, _, _, _ = simular(r_t)
    custos_testes.append(c_tot)

fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.plot(ns_testes, custos_testes, marker='o', linestyle='-', color='purple')
ax3.axvline(x=nivel_servico_alvo, color='red', linestyle='--', label=f'Decisão Atual do Painel ({nivel_servico_alvo}%)')
ax3.set_xlabel("Meta de Nível de Serviço (%)")
ax3.set_ylabel("Custo Total Logístico (R$)")
ax3.grid(alpha=0.3)
ax3.legend()
st.pyplot(fig3)
