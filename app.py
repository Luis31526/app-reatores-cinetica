import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Isotermas de Adsorção",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do aplicativo
st.title("📊 Análise de Isotermas de Adsorção")
st.markdown("""
Este programa permite analisar e visualizar diferentes modelos de isotermas de adsorção, como Isotermas do Tipo I (Langmuir e Freundlich) e Tipo II (BET).
Carregue um arquivo .csv com duas colunas: Ce e qe para realização dos cálculos e avaliação dos resultados.
""")

# Funções para os modelos de isoterma
def langmuir_isotherm(Ce, Qmax, Kl):
    return (Qmax * Kl * Ce) / (1 + Kl * Ce)


def freundlich_isotherm(Ce, Kf, n):
    return Kf * Ce ** (1/n)


def bet_isotherm(Ce, Qm, C):
    return (Qm * C * Ce) / ((Ce_s - Ce) * (1 + (C - 1) * (Ce / Ce_s)))

# Função para calcular R²
def r_squared(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr**2


# Upload de arquivo
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")

# Parâmetros para isotermas
st.sidebar.header("Parâmetro do Modelo BET")
Ce_s = st.sidebar.number_input(
    "Concentração de saturação (Ce_s)", value=100.0, min_value=0.1)

if uploaded_file is not None:
    # Carregar dados
    df = pd.read_csv(uploaded_file)

    # Verificar se as colunas necessárias existem
    if 'Ce' not in df.columns or 'Qe' not in df.columns:
        st.error("O arquivo CSV deve conter as colunas 'Ce' e 'Qe'")
    else:
        # Criar abas para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Dados e Gráfico",
            "🔍 Langmuir",
            "📊 Freundlich",
            "🔄 BET"
        ])

        with tab1:
            st.header("Dados e Visualização Inicial")

            # Exibir dados
            st.subheader("Dados Carregados")
            st.dataframe(df)

            # Gráfico dos dados experimentais
            st.markdown("---")
            st.subheader("Dados Experimentais")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Ce'],
                y=df['Qe'],
                mode='markers',
                name='Dados Experimentais',
                marker=dict(size=8, color='blue')
            ))

            fig.update_layout(
                title='Isoterma de Adsorção - Dados Experimentais',
                xaxis_title='Ce',
                yaxis_title='qe',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Estatísticas descritivas
            st.markdown("---")
            st.subheader("Estatísticas Descritivas")
            estatisticas = df.describe()

            # Crie um dicionário com os termos em inglês e suas traduções em português
            traducao = {
                'count': 'Quantidade de dados',
                'mean': 'Média',
                'std': 'Desvio padrão',
                'min': 'Mínimo',
                '25%': '25%',
                '50%': 'Mediana (50%)',
                '75%': '75%',
                'max': 'Máximo'
            }

            # Renomeia o índice do dataframe de estatísticas usando o dicionário e traduz para pt-br
            estatisticas_traduzidas = estatisticas.rename(index=traducao)
            st.dataframe(estatisticas_traduzidas)

        with tab2:
            st.header("Isoterma de Langmuir")

            # Ajuste do modelo de Langmuir
            try:
                popt_langmuir, pcov_langmuir = curve_fit(
                    langmuir_isotherm, df['Ce'], df['Qe'], p0=[max(df['Qe']), 0.1])
                Qmax, Kl = popt_langmuir
                df['Langmuir'] = langmuir_isotherm(df['Ce'], Qmax, Kl)
                r2_langmuir = r_squared(df['Qe'], df['Langmuir'])

                # Exibe parâmetros
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Qmax", f"{Qmax:.4f}")
                with col2:
                    st.metric("Kl", f"{Kl:.4f}")
                with col3:
                    st.metric("R²", f"{r2_langmuir:.4f}")

                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Ce'],
                    y=df['Qe'],
                    mode='markers',
                    name='Dados Experimentais',
                    marker=dict(size=8, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=df['Ce'],
                    y=df['Langmuir'],
                    mode='lines',
                    name='Modelo Langmuir',
                    line=dict(color='red', width=3)
                ))
                fig.update_layout(
                    title='Ajuste do Modelo de Langmuir',
                    xaxis_title='Ce (Concentração de Equilíbrio)',
                    yaxis_title='Qe (Capacidade de Adsorção)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Linearização de Langmuir
                st.markdown("---")
                st.subheader("Linearização de Langmuir")
                df['1/Ce'] = 1 / df['Ce']
                df['1/Qe'] = 1 / df['Qe']

                fig_lin = go.Figure()
                fig_lin.add_trace(go.Scatter(
                    x=df['1/Ce'],
                    y=df['1/Qe'],
                    mode='markers',
                    name='Dados Linearizados',
                    marker=dict(size=8, color='green')
                ))

                # Ajuste linear
                try:
                    from sklearn.linear_model import LinearRegression
                    X = df['1/Ce'].values.reshape(-1, 1)
                    y = df['1/Qe'].values
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)

                    fig_lin.add_trace(go.Scatter(
                        x=df['1/Ce'],
                        y=y_pred,
                        mode='lines',
                        name='Ajuste Linear',
                        line=dict(color='orange', width=3)
                    ))

                    r2_linear = model.score(X, y)
                    st.metric("R² da linearização", f"{r2_linear:.4f}")

                except Exception as e:
                    st.warning(
                        f"Não foi possível realizar a linearização: {e}")

                fig_lin.update_layout(
                    title='Linearização de Langmuir (1/Qe vs 1/Ce)',
                    xaxis_title='1/Ce',
                    yaxis_title='1/Qe',
                    height=500
                )
                st.plotly_chart(fig_lin, use_container_width=True)

            except Exception as e:
                st.error(f"Erro no ajuste de Langmuir: {e}")

        with tab3:
            st.header("Isoterma de Freundlich")

            # Ajusta o modelo de Freundlich
            try:
                popt_freundlich, pcov_freundlich = curve_fit(
                    freundlich_isotherm, df['Ce'], df['Qe'],
                    p0=[1, 1], bounds=([0, 0.1], [1000, 10])
                )
                Kf, n = popt_freundlich
                df['Freundlich'] = freundlich_isotherm(df['Ce'], Kf, n)
                r2_freundlich = r_squared(df['Qe'], df['Freundlich'])

                # Exibe parâmetros
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Kf", f"{Kf:.4f}")
                with col2:
                    st.metric("n", f"{n:.4f}")
                with col3:
                    st.metric("R²", f"{r2_freundlich:.4f}")

                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Ce'],
                    y=df['Qe'],
                    mode='markers',
                    name='Dados Experimentais',
                    marker=dict(size=8, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=df['Ce'],
                    y=df['Freundlich'],
                    mode='lines',
                    name='Modelo Freundlich',
                    line=dict(color='purple', width=3)
                ))
                fig.update_layout(
                    title='Ajuste do Modelo de Freundlich',
                    xaxis_title='Ce (Concentração de Equilíbrio)',
                    yaxis_title='Qe (Capacidade de Adsorção)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Linearização de Freundlich
                st.markdown("---")
                st.subheader("Linearização de Freundlich")
                df['log_Ce'] = np.log10(df['Ce'])
                df['log_Qe'] = np.log10(df['Qe'])

                fig_lin = go.Figure()
                fig_lin.add_trace(go.Scatter(
                    x=df['log_Ce'],
                    y=df['log_Qe'],
                    mode='markers',
                    name='Dados Linearizados',
                    marker=dict(size=8, color='green')
                ))

                # Ajuste linear
                try:
                    X = df['log_Ce'].values.reshape(-1, 1)
                    y = df['log_Qe'].values
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)

                    fig_lin.add_trace(go.Scatter(
                        x=df['log_Ce'],
                        y=y_pred,
                        mode='lines',
                        name='Ajuste Linear',
                        line=dict(color='orange', width=3)
                    ))

                    r2_linear = model.score(X, y)
                    st.metric("R² da linearização", f"{r2_linear:.4f}")

                except Exception as e:
                    st.warning(
                        f"Não foi possível realizar a linearização: {e}")

                fig_lin.update_layout(
                    title='Linearização de Freundlich (log Qe vs log Ce)',
                    xaxis_title='log Ce',
                    yaxis_title='log Qe',
                    height=500
                )
                st.plotly_chart(fig_lin, use_container_width=True)

            except Exception as e:
                st.error(f"Erro no ajuste de Freundlich: {e}")

        with tab4:
            st.header("Isoterma de BET")

            # Ajuste do modelo de BET
            try:
                # Prepara dados para BET (evitar divisão por zero)
                df_bet = df[df['Ce'] < Ce_s].copy()
                df_bet = df_bet[df_bet['Ce'] > 0].copy()

                if len(df_bet) > 3:
                    # Define função BET para ajuste
                    def bet_isotherm_fit(Ce, Qm, C):
                        return (Qm * C * Ce) / ((Ce_s - Ce) * (1 + (C - 1) * (Ce / Ce_s)))

                    popt_bet, pcov_bet = curve_fit(
                        bet_isotherm_fit, df_bet['Ce'], df_bet['Qe'],
                        p0=[max(df_bet['Qe']), 100], bounds=([0, 1], [1000, 1000])
                    )
                    Qm, C = popt_bet
                    df_bet['BET'] = bet_isotherm_fit(df_bet['Ce'], Qm, C)
                    r2_bet = r_squared(df_bet['Qe'], df_bet['BET'])

                    # Exibe parâmetros
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Qm", f"{Qm:.4f}")
                    with col2:
                        st.metric("C", f"{C:.4f}")
                    with col3:
                        st.metric("Ce_s", f"{Ce_s:.2f}")
                    with col4:
                        st.metric("R²", f"{r2_bet:.4f}")

                    # Gráfico
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_bet['Ce'],
                        y=df_bet['Qe'],
                        mode='markers',
                        name='Dados Experimentais',
                        marker=dict(size=8, color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_bet['Ce'],
                        y=df_bet['BET'],
                        mode='lines',
                        name='Modelo BET',
                        line=dict(color='teal', width=3)
                    ))
                    fig.add_vline(x=Ce_s, line_dash="dash", line_color="red",
                                  annotation_text=f"Ce_s = {Ce_s}", annotation_position="top right")
                    fig.update_layout(
                        title='Ajuste do Modelo de BET',
                        xaxis_title='Ce (Concentração de Equilíbrio)',
                        yaxis_title='Qe (Capacidade de Adsorção)',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Linearização de BET
                    st.markdown("---")
                    st.subheader("Linearização de BET")
                    df_bet['Ce/Ce_s'] = df_bet['Ce'] / Ce_s
                    df_bet['X'] = df_bet['Ce'] / \
                        (df_bet['Qe'] * (Ce_s - df_bet['Ce']))

                    fig_lin = go.Figure()
                    fig_lin.add_trace(go.Scatter(
                        x=df_bet['Ce/Ce_s'],
                        y=df_bet['X'],
                        mode='markers',
                        name='Dados Linearizados',
                        marker=dict(size=8, color='green')
                    ))

                    # Ajuste linear
                    try:
                        X = df_bet['Ce/Ce_s'].values.reshape(-1, 1)
                        y = df_bet['X'].values
                        model = LinearRegression().fit(X, y)
                        y_pred = model.predict(X)

                        fig_lin.add_trace(go.Scatter(
                            x=df_bet['Ce/Ce_s'],
                            y=y_pred,
                            mode='lines',
                            name='Ajuste Linear',
                            line=dict(color='orange', width=3)
                        ))

                        r2_linear = model.score(X, y)
                        st.metric("R² da linearização", f"{r2_linear:.4f}")

                    except Exception as e:
                        st.warning(
                            f"Não foi possível realizar a linearização: {e}")

                    fig_lin.update_layout(
                        title='Linearização de BET',
                        xaxis_title='Ce/Ce_s',
                        yaxis_title='Ce/(Qe*(Ce_s-Ce))',
                        height=500
                    )
                    st.plotly_chart(fig_lin, use_container_width=True)

                else:
                    st.warning(
                        "Dados insuficientes para ajuste do modelo BET. Verifique Ce_s ou adicione mais pontos próximos à saturação.")

            except Exception as e:
                st.error(f"Erro no ajuste de BET: {e}")

        # Comparação entre modelos
        st.markdown("---")
        st.header("📋 Comparação entre Modelos")

        # Criar tabela de comparação
        comparison_data = []

        try:
            comparison_data.append(
                {"Modelo": "Langmuir", "R²": f"{r2_langmuir:.4f}"})
        except:
            comparison_data.append({"Modelo": "Langmuir", "R²": "N/A"})

        try:
            comparison_data.append(
                {"Modelo": "Freundlich", "R²": f"{r2_freundlich:.4f}"})
        except:
            comparison_data.append({"Modelo": "Freundlich", "R²": "N/A"})

        try:
            comparison_data.append({"Modelo": "BET", "R²": f"{r2_bet:.4f}"})
        except:
            comparison_data.append({"Modelo": "BET", "R²": "N/A"})

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)

        # Gráfico comparativo
        st.subheader("Comparação Visual dos Modelos")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=df['Ce'],
            y=df['Qe'],
            mode='markers',
            name='Dados Experimentais',
            marker=dict(size=8, color='black')
        ))

        try:
            fig_comp.add_trace(go.Scatter(
                x=df['Ce'],
                y=df['Langmuir'],
                mode='lines',
                name='Langmuir',
                line=dict(color='red', width=2)
            ))
        except:
            pass

        try:
            fig_comp.add_trace(go.Scatter(
                x=df['Ce'],
                y=df['Freundlich'],
                mode='lines',
                name='Freundlich',
                line=dict(color='blue', width=2)
            ))
        except:
            pass

        try:
            fig_comp.add_trace(go.Scatter(
                x=df_bet['Ce'],
                y=df_bet['BET'],
                mode='lines',
                name='BET',
                line=dict(color='purple', width=2)
            ))
        except:
            pass

        fig_comp.update_layout(
            title='Comparação entre Modelos de Isotermas',
            xaxis_title='Ce (Concentração de Equilíbrio)',
            yaxis_title='Qe (Capacidade de Adsorção)',
            height=500
        )
        st.plotly_chart(fig_comp, use_container_width=True)


else:
    st.info("Por favor, faça upload de um arquivo CSV para começar.")

    # Exemplo de dados
    st.subheader("Exemplo de formato do arquivo CSV")
    exemplo_df = pd.DataFrame({
        'Ce': [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        'Qe': [0.5, 2.1, 3.8, 6.5, 8.2, 9.3, 9.8]
    })
    st.dataframe(exemplo_df)

# Rodapé
st.markdown("---")
st.caption("Aplicativo desenvolvido para análise de isotermas de adsorção")
st.caption("Desenvolvido por Luis Henrique")
st.caption("Engenharia Química - Universidade Federal do Amazonas (UFAM)")
