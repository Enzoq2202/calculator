import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import logging
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

def main():
    """Página principal para entrada de parâmetros e cálculo de preços."""
    st.set_page_config(
        page_title="Calculadora de Opções",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    # 🎉 Calculadora de Opções
    Este aplicativo permite calcular preços de **Opções Europeias** e **Opções Asiáticas** 
    usando simulação Monte Carlo. Use a barra lateral para configurar os parâmetros e 
    navegue até a página "Simulações Monte Carlo" para visualizar as trajetórias.
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
    ) / 100

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

                # Armazenar parâmetros no session_state para a página Monte Carlo
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

                # Apresentação dos resultados
                st.subheader("📊 Resultados")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Preço Atual (S0)", f"$ {S0:.2f}")
                col2.metric("Strike (K)", f"$ {K:.2f}")
                col3.metric("Volatilidade (σ)", f"{sigma:.2%}")
                col4.metric(f"Preço da {option_type}", f"$ {price:.2f}")

                # Gráfico de histórico de preço
                st.subheader("📈 Histórico de Preço")
                fig_hist = plt.figure(figsize=(10, 4))
                ax_hist = fig_hist.add_subplot(111)
                dados['Close'].plot(ax=ax_hist)
                ax_hist.set_title(f"Fechamento Diário - {ticker}")
                ax_hist.set_xlabel("Data")
                ax_hist.set_ylabel("Preço ($)")
                st.pyplot(fig_hist)
                plt.close(fig_hist)
                st.success("Cálculos concluídos! Acesse a página 'Simulações Monte Carlo' na barra lateral para ver as trajetórias e ajustar o horizonte temporal, se desejar.")

        except Exception as e:
            st.error(f"Erro ao processar: {str(e)}. Verifique o ticker ou a conexão com a internet.")
            logger.error(f"Erro na execução: {str(e)}")

if __name__ == "__main__":
    main()