import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import logging
import seaborn as sns
import re

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def capturar_parametros(ticker: str, periodo: str = '1y') -> Tuple[float, float, float, pd.DataFrame]:
    """
    Captura parâmetros do ativo usando dados históricos do Yahoo Finance.
    
    Args:
        ticker: Símbolo da ação (ex: AAPL, ITUB3.SA)
        periodo: Período de dados históricos (default: '1y')
    
    Returns:
        Tuple contendo (preço atual, retorno médio, volatilidade, dados históricos)
    """
    # Validação básica do ticker
    if not ticker or not re.match(r'^[A-Z0-9.]+$', ticker):
        raise ValueError("Ticker inválido. Use formato como 'AAPL' ou 'ITUB3.SA'.")
    
    try:
        dados = yf.download(ticker, period=periodo)
        if dados.empty:
            raise ValueError(f"Nenhum dado encontrado para o ticker {ticker}")
            
        dados['Retornos'] = dados['Close'].pct_change().apply(np.log1p)
        dados = dados.dropna()
        
        S0 = dados['Close'].iloc[-1].item()
        mu = dados['Retornos'].mean() * 252
        sigma = dados['Retornos'].std() * np.sqrt(252)
        
        return S0, mu, sigma, dados
    except Exception as e:
        logger.error(f"Erro ao capturar parâmetros: {str(e)}")
        raise

def monte_carlo_opcao_europeia(
    S0: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    n_sim: int = 10000
) -> float:
    """
    Calcula o preço de uma opção europeia usando simulação Monte Carlo.
    
    Args:
        S0: Preço atual do ativo
        K: Preço de exercício
        T: Tempo até expiração (em anos)
        r: Taxa livre de risco
        sigma: Volatilidade
        n_sim: Número de simulações
    
    Returns:
        Preço estimado da opção
    """
    Z = np.random.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def monte_carlo_opcao_asiatica(
    S0: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    n_sim: int = 10000, 
    n_steps: int = 252
) -> float:
    """
    Calcula o preço de uma opção asiática usando simulação Monte Carlo.
    
    Args:
        S0: Preço atual do ativo
        K: Preço de exercício
        T: Tempo até expiração (em anos)
        r: Taxa livre de risco
        sigma: Volatilidade
        n_sim: Número de simulações
        n_steps: Número de passos na simulação
    
    Returns:
        Preço estimado da opção
    """
    dt = T / n_steps
    Z = np.random.normal(size=(n_sim, n_steps))
    prices = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))
    media = np.mean(prices, axis=1)
    payoffs = np.maximum(media - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)

@st.cache_data
def plot_trajetorias(
    S0: float, 
    T: float, 
    r: float, 
    sigma: float, 
    n_sim: int = 100, 
    n_steps: int = 252
) -> plt.Figure:
    """
    Gera um gráfico moderno com simulações de trajetórias de preço.
    Inclui design aprimorado com gradientes, anotações e estilo visual moderno.
    
    Args:
        S0: Preço inicial do ativo
        T: Tempo até expiração (em anos)
        r: Taxa livre de risco
        sigma: Volatilidade
        n_sim: Número de simulações
        n_steps: Número de passos na simulação
    
    Returns:
        Figura Matplotlib com as trajetórias plotadas
    """
    plt.style.use('seaborn-v0_8')  # Tema atualizado
    sns.set_context("notebook", font_scale=1.2)
    
    dt = T / n_steps
    n_steps = int(T * 252)  # Ajustar passos proporcionalmente ao tempo
    trajetorias = []
    
    for _ in range(min(n_sim, 50)):
        prices = [S0]
        for _ in range(n_steps):
            Z = np.random.standard_normal()
            St = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            prices.append(St)
        trajetorias.append(prices)
    
    trajetorias = np.array(trajetorias)
    media = trajetorias.mean(axis=0)
    
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)
    
    for i, traj in enumerate(trajetorias):
        ax.plot(traj, color=sns.color_palette("Blues", n_colors=50)[min(i, 49)], 
                alpha=0.15, linewidth=1.2)
    
    ax.plot(media, color='#e67e22', linewidth=3, label='Média das Trajetórias', zorder=10)
    from matplotlib.patheffects import withStroke
    ax.get_lines()[-1].set_path_effects([withStroke(linewidth=5, foreground='black', alpha=0.2)])
    
    ax.axhline(S0, color='gray', linestyle='--', alpha=0.5, label=f'Preço Inicial ($ {S0:.2f})')
    
    ax.set_title('Simulação Monte Carlo - Trajetórias de Preço', 
                 fontsize=18, pad=20, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Dias', fontsize=14, color='#34495e')
    ax.set_ylabel('Preço ($)', fontsize=14, color='#34495e')
    
    ax.grid(True, linestyle='--', alpha=0.4, color='#bdc3c7')
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, 
              facecolor='white', edgecolor='#ecf0f1', fontsize=12)
    
    ax.annotate(f'Média Final: ${media[-1]:.2f}', 
                xy=(n_steps, media[-1]), 
                xytext=(n_steps-50, media[-1]+S0*0.05),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=12, color='#2c3e50', bbox=dict(facecolor='white', alpha=0.8))
    
    ax.margins(y=0.1)
    fig.tight_layout()
    
    return fig

def main():
    """Função principal que configura e executa a interface Streamlit."""
    st.set_page_config(
        page_title="Calculadora de Opções", 
        page_icon="📈", 
        layout="wide"
    )

    st.markdown("""
    # 🎉 Calculadora de Opções
    Este aplicativo permite calcular preços de **Opções Europeias** e **Opções Asiáticas** 
    usando simulação Monte Carlo.
    """)

    # Sidebar com configurações
    st.sidebar.header("⚙️ Configurações")
    periodo = st.sidebar.selectbox(
        "Período de dados históricos",
        ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    n_sim = st.sidebar.slider(
        "Número de simulações",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )

    r = st.sidebar.number_input(
        "Taxa livre de risco (% ao ano)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.1
    ) / 100  # Converter para decimal

    # Seleção de tipo de opção e ticker
    option_type = st.radio(
        "Selecione o tipo de opção:",
        ("Opção Europeia", "Opção Asiática")
    )

    ticker = st.text_input(
        "Digite o ticker da ação (ex: AAPL para EUA, ITUB3.SA para Brasil):",
        value="AAPL"
    )

    # Parâmetros da opção
    col1, col2 = st.columns(2)
    with col1:
        strike_percent = st.slider(
            "Strike (% acima do preço atual)",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        )
    
    with col2:
        tempo_expiracao = st.slider(
            "Tempo até expiração (anos)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )

    if st.button("Calcular"):
        try:
            with st.spinner("Calculando..."):
                S0, mu, sigma, dados = capturar_parametros(ticker, periodo)
                K = S0 * (1 + strike_percent/100)
                T = tempo_expiracao

                if option_type == "Opção Europeia":
                    price = monte_carlo_opcao_europeia(S0, K, T, r, sigma, n_sim)
                else:
                    price = monte_carlo_opcao_asiatica(S0, K, T, r, sigma, n_sim)

                # Apresentação dos resultados
                st.subheader("📊 Resultados")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Preço Atual (S0)", f"$ {S0:.2f}")
                col2.metric("Strike (K)", f"$ {K:.2f}")
                col3.metric("Volatilidade (σ)", f"{sigma:.2%}")
                col4.metric(f"Preço da {option_type}", f"$ {price:.2f}")

                # Gráfico de histórico de preço
                st.subheader("📈 Histórico de Preço")
                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                dados['Close'].plot(ax=ax_hist)
                ax_hist.set_title(f"Fechamento Diário - {ticker}")
                ax_hist.set_xlabel("Data")
                ax_hist.set_ylabel("Preço ($)")
                st.pyplot(fig_hist)

                # Simulação de trajetórias
                if st.checkbox("Mostrar simulações Monte Carlo"):
                    st.subheader("🚀 Simulações Monte Carlo")
                    st.write("Apenas 50 trajetórias são exibidas para melhor visualização.")
                    fig_sim = plot_trajetorias(S0, T, r, sigma, n_sim=min(n_sim, 100))
                    st.pyplot(fig_sim)
    
        except Exception as e:
            st.error(f"Erro ao processar: {str(e)}. Verifique o ticker ou a conexão com a internet.")
            logger.error(f"Erro na execução: {str(e)}")

if __name__ == "__main__":
    main()