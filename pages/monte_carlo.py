import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    plt.style.use('default')  # Usar tema padrão e personalizar manualmente
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
    
    # Ajustar margens e layout
    ax.margins(y=0.1)
    fig.tight_layout()
    
    return fig

def main():
    """Página para exibir simulações Monte Carlo."""
    st.title("🚀 Simulações Monte Carlo")
    st.markdown("Visualize as trajetórias simuladas do preço do ativo. Ajuste o horizonte temporal e o número de trajetórias exibidas abaixo.")

    # Verificar se os parâmetros estão no session_state
    if 'S0' not in st.session_state:
        st.warning("Por favor, calcule os parâmetros na página principal antes de visualizar as simulações.")
        return

    try:
        # Recuperar parâmetros do session_state
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