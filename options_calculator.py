import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
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

def plot_trajetorias(
    S0: float, 
    T: float, 
    r: float, 
    sigma: float, 
    n_sim: int = 100, 
    n_traj: int = 20,
    n_steps: int = 252
) -> plt.Figure:
    """
    Gera um gráfico moderno e limpo com simulações de trajetórias de preço.
    
    Args:
        S0: Preço inicial do ativo
        T: Tempo até expiração (em anos)
        r: Taxa livre de risco
        sigma: Volatilidade
        n_sim: Número total de simulações
        n_traj: Número de trajetórias a exibir
        n_steps: Número de passos na simulação
    
    Returns:
        Figura Matplotlib com as trajetórias plotadas
    """
    logger.info("Gerando gráfico Monte Carlo...")
    plt.style.use('default')
    sns.set_context("notebook", font_scale=1.2)
    
    dt = T / n_steps
    n_steps = int(T * 252)
    paths = np.zeros((n_steps + 1, n_sim))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_sim)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plotar trajetórias com uma paleta de cores mais suave
    for i in range(min(n_sim, n_traj)):
        ax.plot(paths[:, i], color=sns.color_palette("cool", n_colors=n_traj)[i], 
                alpha=0.3, linewidth=0.8)
    
    # Plotar média das trajetórias
    media = paths.mean(axis=1)
    ax.plot(media, color='#FF6F61', linewidth=2.5, label='Média das Trajetórias', zorder=10)
    from matplotlib.patheffects import withStroke
    ax.get_lines()[-1].set_path_effects([withStroke(linewidth=4, foreground='black', alpha=0.15)])
    
    # Linha do preço inicial
    ax.axhline(S0, color='#B0BEC5', linestyle='--', alpha=0.6, label=f'Preço Inicial ($ {S0:.2f})')
    
    # Configurações visuais modernas
    ax.set_title('Simulação Monte Carlo - Trajetórias de Preço', 
                 fontsize=16, pad=15, fontweight='medium', color='#263238')
    ax.set_xlabel('Dias', fontsize=12, color='#455A64')
    ax.set_ylabel('Preço ($)', fontsize=12, color='#455A64')
    
    # Grid suave
    ax.grid(True, linestyle='--', alpha=0.2, color='#CFD8DC')
    
    # Legenda sem borda
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    
    # Anotação para preço final médio com fundo suave
    ax.annotate(f'Média Final: ${media[-1]:.2f}', 
                xy=(n_steps, media[-1]), 
                xytext=(n_steps-50, media[-1]+S0*0.05),
                arrowprops=dict(facecolor='#455A64', arrowstyle='->', linewidth=1),
                fontsize=10, color='#263238', 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax.margins(y=0.1)
    fig.tight_layout()
    
    return fig

def main():
    """Função principal que alterna entre as páginas Calculadora e Simulações."""
    st.set_page_config(
        page_title="Calculadora de Opções",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Navegação por rádio botão na sidebar
    page = st.sidebar.radio(
        "Navegação",
        ["Calculadora de Opções", "Simulações Monte Carlo"],
        index=0
    )

    # Configurações comuns na sidebar
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
    ) / 100

    ticker = st.sidebar.text_input(
        "Digite o ticker da ação (ex: AAPL, ITUB3.SA):",
        value="AAPL"
    )

    # Página Calculadora de Opções
    if page == "Calculadora de Opções":
        st.markdown("""
        # 🎉 Calculadora de Opções
        Este aplicativo permite calcular preços de **Opções Europeias** e **Opções Asiáticas** 
        usando simulação Monte Carlo.
        """)

        option_type = st.radio(
            "Selecione o tipo de opção:",
            ("Opção Europeia", "Opção Asiática")
        )

        col1, col2 = st.columns(2)
        with col1:
            strike_percent = st.slider(
                "Strike (% acima/abaixo do preço atual)",
                min_value=-50,
                max_value=50,
                value=5,
                step=1,
                help="Valores negativos indicam strike abaixo do preço atual"
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

                    # Armazenar parâmetros no session_state
                    st.session_state['S0'] = S0
                    st.session_state['mu'] = mu
                    st.session_state['sigma'] = sigma
                    st.session_state['dados'] = dados
                    st.session_state['K'] = K
                    st.session_state['T'] = T
                    st.session_state['r'] = r
                    st.session_state['n_sim'] = n_sim
                    st.session_state['ticker'] = ticker
                    st.session_state['option_type'] = option_type

                    if option_type == "Opção Europeia":
                        price = monte_carlo_opcao_europeia(S0, K, T, r, sigma, n_sim)
                    else:
                        price = monte_carlo_opcao_asiatica(S0, K, T, r, sigma, n_sim)

                    st.subheader("📊 Resultados")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Preço Atual (S0)", f"$ {S0:.2f}")
                    col2.metric("Strike (K)", f"$ {K:.2f}")
                    col3.metric("Volatilidade (σ)", f"{sigma:.2%}")
                    col4.metric(f"Preço da {option_type}", f"$ {price:.2f}")

                    st.subheader("📈 Histórico de Preço")
                    fig_hist = plt.figure(figsize=(10, 4))
                    ax_hist = fig_hist.add_subplot(111)
                    dados['Close'].plot(ax=ax_hist)
                    ax_hist.set_title(f"Fechamento Diário - {ticker}")
                    ax_hist.set_xlabel("Data")
                    ax_hist.set_ylabel("Preço ($)")
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
                    st.success("Cálculos concluídos! Acesse 'Simulações Monte Carlo' para visualizar as trajetórias.")

            except Exception as e:
                st.error(f"Erro ao processar: {str(e)}. Verifique o ticker ou a conexão com a internet.")
                logger.error(f"Erro na execução: {str(e)}")

    # Página Simulações Monte Carlo
    elif page == "Simulações Monte Carlo":
        st.title("🚀 Simulações Monte Carlo")
        st.markdown("Visualize as trajetórias simuladas do preço do ativo. Ajuste o horizonte temporal e o número de trajetórias abaixo.")

        if 'S0' not in st.session_state:
            st.warning("Por favor, calcule os parâmetros na página 'Calculadora de Opções' antes de visualizar as simulações.")
            return

        try:
            S0 = st.session_state['S0']
            r = st.session_state['r']
            sigma = st.session_state['sigma']
            n_sim = st.session_state['n_sim']
            ticker = st.session_state['ticker']

            # Slider para ajustar o horizonte temporal
            T = st.slider(
                "Tempo até expiração (anos)",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state['T'],
                step=0.1,
                help="Ajuste o horizonte temporal da simulação (em anos)."
            )
            st.session_state['T'] = T  # Atualizar o session_state

            # Slider para ajustar o número de trajetórias exibidas
            n_traj = st.slider(
                "Número de trajetórias exibidas",
                min_value=1,
                max_value=50,
                value=20,
                step=1,
                help="Escolha quantas trajetórias deseja exibir no gráfico (máximo 50)."
            )

            st.write(f"Exibindo simulações para o ticker: **{ticker}**")
            st.write(f"Exibindo {n_traj} de {min(n_sim, 100)} trajetórias simuladas.")

            with st.spinner("Gerando simulações Monte Carlo..."):
                fig_sim = plot_trajetorias(S0, T, r, sigma, n_sim=min(n_sim, 100), n_traj=n_traj)
                st.pyplot(fig_sim)
                plt.close(fig_sim)
                logger.info("Gráfico Monte Carlo renderizado com sucesso.")
                st.success("Simulações Monte Carlo geradas com sucesso!")

        except Exception as e:
            st.error(f"Erro ao gerar simulações: {str(e)}")
            logger.error(f"Erro na renderização do gráfico Monte Carlo: {str(e)}")

if __name__ == "__main__":
    main()