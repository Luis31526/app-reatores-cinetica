import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Simulador de Coluna de Absorção",
    page_icon=":test_tube:",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .stSelectbox>div>div>div {
        color: #333;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# Título e descrição
st.title("📊 Simulador de Coluna de Absorção")
st.markdown("""
Este simulador demonstra a operação de uma coluna de absorção para separação de gases. 
Ajuste os parâmetros ou importe seus próprios dados para personalizar a simulação.
""")

# Sidebar para parâmetros de entrada
with st.sidebar:
    st.header("⚙️ Parâmetros da Simulação")

    # Opção para usar dados padrão ou importar
    data_option = st.radio("Fonte de dados:",
                           ("Usar dados padrão", "Importar dados personalizados"))

    if data_option == "Usar dados padrão":
        st.info("Utilizando parâmetros padrão da simulação.")

        # Parâmetros padrão
        column_height = st.slider("Altura da coluna (m)", 1.0, 10.0, 5.0, 0.1)
        diameter = st.slider("Diâmetro da coluna (m)", 0.1, 2.0, 0.5, 0.05)
        gas_flow = st.slider("Vazão do gás (kg/h)", 10.0, 500.0, 100.0, 5.0)
        liquid_flow = st.slider(
            "Vazão do líquido (kg/h)", 10.0, 500.0, 150.0, 5.0)
        inlet_concentration = st.slider(
            "Concentração de entrada (ppm)", 100, 10000, 1000, 50)
        efficiency = st.slider("Eficiência da coluna (%)", 50, 100, 85, 1)

    else:
        st.info("Por favor, importe seus dados personalizados.")
        uploaded_file = st.file_uploader("Carregar arquivo CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Dados carregados com sucesso!")
                st.dataframe(df.head())

                # Extrair parâmetros do arquivo (se existirem)
                if 'column_height' in df.columns:
                    column_height = float(df['column_height'].iloc[0])
                else:
                    column_height = 5.0

                if 'diameter' in df.columns:
                    diameter = float(df['diameter'].iloc[0])
                else:
                    diameter = 0.5

                # Adicione mais parâmetros conforme necessário

            except Exception as e:
                st.error(f"Erro ao ler o arquivo: {e}")
                st.info("Usando parâmetros padrão como fallback.")
                column_height = 5.0
                diameter = 0.5
                gas_flow = 100.0
                liquid_flow = 150.0
                inlet_concentration = 1000
                efficiency = 85
        else:
            st.warning(
                "Por favor, carregue um arquivo CSV ou use dados padrão.")
            st.stop()

# Conteúdo principal
tab1, tab2, tab3 = st.tabs(["Simulação", "Gráficos", "Resultados"])

with tab1:
    st.header("Configuração da Simulação")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parâmetros Físicos")
        st.metric("Altura da Coluna", f"{column_height} m")
        st.metric("Diâmetro da Coluna", f"{diameter} m")
        st.metric("Área da Seção Transversal",
                  f"{np.pi * (diameter/2)**2:.3f} m²")

    with col2:
        st.subheader("Parâmetros de Operação")
        st.metric("Vazão do Gás", f"{gas_flow} kg/h")
        st.metric("Vazão do Líquido", f"{liquid_flow} kg/h")
        st.metric("Concentração de Entrada", f"{inlet_concentration} ppm")
        st.metric("Eficiência", f"{efficiency}%")

    st.markdown("---")

    # Visualização da coluna
    st.subheader("Visualização da Coluna de Absorção")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Desenhar a coluna
    ax.plot([0, 0], [0, column_height], 'k-', linewidth=10)  # Parede esquerda
    ax.plot([1, 1], [0, column_height], 'k-', linewidth=10)  # Parede direita

    # Adicionar preenchimento para o líquido
    ax.fill_between([0, 1], 0, column_height, color='lightblue', alpha=0.3)

    # Adicionar bolhas para representar o gás
    for i in range(20):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, column_height-0.1)
        size = np.random.uniform(5, 20)
        ax.scatter(x, y, s=size, c='red', alpha=0.5)

    # Configurações do gráfico
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, column_height + 0.5)
    ax.set_title("Representação da Coluna de Absorção")
    ax.set_xlabel("Largura")
    ax.set_ylabel("Altura (m)")
    ax.grid(False)
    ax.set_xticks([])

    st.pyplot(fig)

with tab2:
    st.header("Resultados Gráficos")

    # Simular dados de concentração ao longo da coluna
    heights = np.linspace(0, column_height, 50)
    concentrations = inlet_concentration * np.exp(-efficiency/100 * heights)

    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Perfil de concentração
    ax1.plot(concentrations, heights, 'b-', linewidth=2)
    ax1.set_title("Perfil de Concentração ao Longo da Coluna")
    ax1.set_xlabel("Concentração (ppm)")
    ax1.set_ylabel("Altura (m)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Gráfico 2: Eficiência de remoção
    removal_efficiency = 100 * (1 - concentrations/inlet_concentration)
    ax2.plot(heights, removal_efficiency, 'r-', linewidth=2)
    ax2.set_title("Eficiência de Remoção ao Longo da Coluna")
    ax2.set_xlabel("Altura (m)")
    ax2.set_ylabel("Eficiência de Remoção (%)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 100)

    st.pyplot(fig)

    # Gráfico 3D opcional
    if st.checkbox("Mostrar visualização 3D (pode ser lento)"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Criar malha para superfície
        X, Y = np.meshgrid(np.linspace(0, diameter, 20),
                           np.linspace(0, column_height, 20))
        Z = inlet_concentration * np.exp(-efficiency/100 * Y)

        # Plotar superfície
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        # Configurações
        ax.set_title("Distribuição 3D de Concentração")
        ax.set_xlabel("Diâmetro (m)")
        ax.set_ylabel("Altura (m)")
        ax.set_zlabel("Concentração (ppm)")
        fig.colorbar(surf, shrink=0.5, aspect=5)

        st.pyplot(fig)

with tab3:
    st.header("Resultados Numéricos")

    # Calcular resultados
    outlet_concentration = inlet_concentration * \
        np.exp(-efficiency/100 * column_height)
    removal_percentage = 100 * (1 - outlet_concentration/inlet_concentration)
    residence_time = column_height * np.pi * (diameter/2)**2 / (gas_flow/3600)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resultados Principais")
        st.metric("Concentração de Saída", f"{outlet_concentration:.2f} ppm")
        st.metric("Eficiência de Remoção", f"{removal_percentage:.1f}%")
        st.metric("Tempo de Residência", f"{residence_time:.2f} segundos")

    with col2:
        st.subheader("Dados para Exportação")

        # Criar DataFrame com resultados
        results_df = pd.DataFrame({
            'Parâmetro': ['Altura da Coluna', 'Diâmetro', 'Vazão do Gás', 'Vazão do Líquido',
                          'Concentração de Entrada', 'Eficiência', 'Concentração de Saída',
                          'Eficiência de Remoção', 'Tempo de Residência'],
            'Valor': [column_height, diameter, gas_flow, liquid_flow, inlet_concentration,
                      efficiency, outlet_concentration, removal_percentage, residence_time],
            'Unidade': ['m', 'm', 'kg/h', 'kg/h', 'ppm', '%', 'ppm', '%', 's']
        })

        st.dataframe(results_df)

        # Botão para download
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download dos Resultados (CSV)",
            data=csv,
            file_name="resultados_coluna_absorcao.csv",
            mime="text/csv"
        )

# Rodapé
st.markdown("---")
st.markdown("""
**Simulador de Coluna de Absorção**  
Desenvolvido por Luis Henrique com Python, Streamlit e Matplotlib  
""")
