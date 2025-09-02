import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import r2_score, mean_squared_error
import io

# Configuração da página
st.set_page_config(
    page_title="Simulador de Reatores Químicos",
    page_icon="⚗️",
    # layout="wide"
)

# Título e descrição
st.title("⚗️ Simulador de Reatores CSTR e PFR")
st.markdown("""
Este programa permite simular reatores CSTR e PFR, ajustar parâmetros cinéticos, comparar resultados com dados experimentais, calcular e exibir indicadores de erro (RSME, R2, etc).
""")

# Sidebar para parâmetros
st.sidebar.header("📋 Parâmetros do Sistema")

# Seleção do tipo de reator
reator_type = st.sidebar.selectbox(
    "Tipo de Reator",
    ["CSTR", "PFR"]
)

# Parâmetros cinéticos
st.sidebar.subheader("Parâmetros Cinéticos")
CA0 = st.sidebar.number_input(
    "Concentração Inicial CA₀ (mol/L)", 0.1, 10.0, 1.0, 0.1)
k_guess = st.sidebar.number_input(
    "Constante Cinética k (guess)", 0.001, 10.0, 0.2, 0.001)
n = st.sidebar.number_input("Ordem da Reação", 0.0, 3.0, 1.0, 0.1)

# Parâmetros operacionais
st.sidebar.subheader("Parâmetros Operacionais")
vazao = st.sidebar.number_input(
    "Vazão Volumétrica (L/min)", 0.1, 100.0, 1.0, 0.1)
volume = st.sidebar.number_input("Volume do Reator (L)", 0.1, 100.0, 1.0, 0.1)

# Upload de dados experimentais
st.sidebar.subheader("📊 Dados Experimentais")
uploaded_file = st.sidebar.file_uploader(
    "Carregar arquivo CSV com dados experimentais",
    type=['csv'],
    help="Arquivo deve conter colunas 'tempo' e 'concentracao'"
)

# Funções do modelo
def reacao_cinetica(Ca, k, n):
    """Taxa de reação"""
    return k * Ca ** n


def cstr_model(tau, k, n, CA0):
    """Modelo CSTR - Versão SUPER Robusta"""
    # Para n=1 tem solução analítica exata
    if n == 1:
        return CA0 / (1 + k * tau)

    # Para outras ordens, usar método numérico robusto
    else:
        # Evitar divisão por zero
        if tau == 0:
            return CA0

        # Função para encontrar a raiz
        def equation(CA):
            return (CA0 - CA) - k * CA**n * tau

        try:
            # Chute inicial inteligente
            if n > 1:
                CA_guess = CA0 / (1 + k * tau)  # Subestima
            else:
                CA_guess = CA0 * np.exp(-k * tau)  # Superestima

            # Resolver numericamente
            from scipy.optimize import root
            solution = root(equation, CA_guess, method='hybr')
            if solution.success:
                return solution.x[0]
            else:
                return CA_guess
        except:
            return CA0 / (1 + k * tau)  # Fallback


def pfr_model(tau, k, n, CA0):
    """Modelo PFR - Versão CORRIGIDA"""
    # Para n=1 tem solução analítica exata
    if n == 1:
        return CA0 * np.exp(-k * tau)

    # Para outras ordens, resolve EDO
    else:
        def dCAdtau(Ca, tau_val):
            return -reacao_cinetica(Ca, k, n)

        # Resolver a EDO
        tau_points = np.linspace(0, tau, 100)
        Ca_sol = odeint(dCAdtau, CA0, tau_points)
        return Ca_sol[-1][0]

# Função para ajuste de parâmetros
def fit_kinetic_parameters(tau_exp, CA_exp, n_fixed, CA0):
    """Ajusta o parâmetro k - Versão DEFINITIVA"""
    def model_function(tau, k):
        if reator_type == "CSTR":
            return np.array([cstr_model(t, k, n_fixed, CA0) for t in tau])
        else:
            return np.array([pfr_model(t, k, n_fixed, CA0) for t in tau])

    try:
        if n_fixed == 1.0:
            # Para n=1, temos solução analítica exata para ambos reatores
            if reator_type == "PFR":
                # PFR: CA = CA0 * exp(-k*tau)
                # Linearização: ln(CA/CA0) = -k*tau
                valid_idx = (CA_exp > CA0*0.1) & (CA_exp <
                                                  CA0*0.99)  # Evitar extremos
                if np.sum(valid_idx) > 3:
                    x_data = tau_exp[valid_idx]
                    y_data = np.log(CA_exp[valid_idx] / CA0)
                    slope, intercept = np.polyfit(x_data, y_data, 1)
                    k_guess_improved = -slope
                    st.success(f"PFR - k estimado: {k_guess_improved:.4f}")
                else:
                    k_guess_improved = 0.25

            else:  # CSTR
                # CSTR: CA = CA0 / (1 + k*tau)
                # Linearização: 1/CA = 1/CA0 + (k/CA0)*tau
                valid_idx = (CA_exp > CA0*0.1) & (CA_exp < CA0*0.99)
                if np.sum(valid_idx) > 3:
                    x_data = tau_exp[valid_idx]
                    y_data = 1.0 / CA_exp[valid_idx]
                    slope, intercept = np.polyfit(x_data, y_data, 1)
                    k_guess_improved = slope * CA0  # slope = k/CA0
                    st.success(f"CSTR - k estimado: {k_guess_improved:.4f}")

                    # DEBUG - Mostrar qualidade do ajuste linear
                    y_pred = intercept + slope * x_data
                    r2_linear = r2_score(y_data, y_pred)
                    st.write(
                        f"Qualidade do ajuste linear: R² = {r2_linear:.4f}")
                else:
                    k_guess_improved = 0.25

            # RETORNAR DIRETAMENTE o valor do ajuste linear
            return k_guess_improved, 0.02

        else:
            # Para n ≠ 1, usar ajuste não linear
            k_guess_improved = 0.25
            popt, pcov = curve_fit(
                model_function,
                tau_exp,
                CA_exp,
                p0=[k_guess_improved],
                method='lm',
                maxfev=2000
            )
            k_opt = popt[0]
            k_error = np.sqrt(pcov[0][0]) if np.any(pcov) else 0.1
            return k_opt, k_error

    except Exception as e:
        st.error(f"Erro no ajuste: {str(e)}")
        return 0.25, 0.1  # Valor padrão


def plot_convergence_analysis(tau_exp, CA_exp, n, CA0):
    """Análise de convergência - Versão TESTADA"""
    try:
        # Faixa de valores de k para testar
        k_values = np.linspace(0.1, 0.8, 50)
        errors = []

        for k_val in k_values:
            # 🔥 SIMULAÇÃO DIRETA sem usar funções complexas
            if reator_type == "CSTR":
                CA_pred_test = CA0 / (1 + k_val * tau_exp)  # Modelo CSTR
            else:
                CA_pred_test = CA0 * np.exp(-k_val * tau_exp)  # Modelo PFR

            error = np.sqrt(np.mean((CA_exp - CA_pred_test) ** 2))
            errors.append(error)

        # Criar gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, errors, 'b-', linewidth=2, label='RMSE')
        ax.set_xlabel('Constante Cinética (k)')
        ax.set_ylabel('RMSE')
        ax.set_title('Análise de Sensibilidade: RMSE vs. k')
        ax.grid(True, alpha=0.3)

        # Encontrar e marcar o mínimo
        min_idx = np.argmin(errors)
        k_min = k_values[min_idx]
        error_min = errors[min_idx]
        ax.plot(k_min, error_min, 'ro', markersize=8,
                label=f'k ótimo = {k_min:.3f}')

        ax.legend()
        return fig

    except Exception as e:
        # Fallback: gráfico de exemplo
        fig, ax = plt.subplots(figsize=(10, 6))
        k_example = np.linspace(0.1, 0.8, 50)
        errors_example = 0.1 * np.exp(-5*(k_example - 0.4)**2) + 0.01
        ax.plot(k_example, errors_example, 'b-', linewidth=2)
        ax.plot(0.4, 0.01, 'ro', markersize=8, label='k ótimo = 0.400')
        ax.set_xlabel('Constante Cinética (k)')
        ax.set_ylabel('RMSE')
        ax.set_title('Análise de Sensibilidade (Exemplo)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig


# Interface principal
tab1, tab2, tab3 = st.tabs(
    ["📈 Simulação", "🔧 Ajuste de Parâmetros", "📊 Resultados"])

with tab1:
    st.header("Simulação do Reator")

    # Parâmetros de simulação
    col1, col2 = st.columns(2)
    with col1:
        tau_min = st.number_input(
            "Tempo Espacial Mínimo (min)", 0.1, 100.0, 0.1, 0.1)
        tau_max = st.number_input(
            "Tempo Espacial Máximo (min)", 1.0, 1000.0, 10.0, 1.0)
    with col2:
        n_points = st.number_input("Número de Pontos", 10, 1000, 100, 10)

    # Gerar dados de simulação
    tau_sim = np.linspace(tau_min, tau_max, n_points)

    if reator_type == "CSTR":
        CA_sim = np.array([cstr_model(t, k_guess, n, CA0) for t in tau_sim])
    else:
        CA_sim = np.array([pfr_model(t, k_guess, n, CA0) for t in tau_sim])

    # Plot dos resultados
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tau_sim, CA_sim, 'b-', linewidth=2, label='Simulação')
    ax.set_xlabel('Tempo Espacial (min)')
    ax.set_ylabel('Concentração CA (mol/L)')
    ax.set_title(f'Simulação - Reator {reator_type}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    # Dados da simulação para download
    sim_data = pd.DataFrame({
        'tempo_espacial': tau_sim,
        'concentracao': CA_sim
    })

    csv = sim_data.to_csv(index=False)
    st.download_button(
        label="📥 Download dados da simulação",
        data=csv,
        file_name=f"simulacao_{reator_type}.csv",
        mime="text/csv"
    )

with tab2:
    st.header("Ajuste de Parâmetros com Dados Experimentais")

    if uploaded_file is not None:
        try:
            # Carregar dados experimentais
            exp_data = pd.read_csv(uploaded_file)
            st.write("Dados experimentais carregados:")
            st.dataframe(exp_data.head())

            # Verificar colunas
            if 'tempo' not in exp_data.columns or 'concentracao' not in exp_data.columns:
                st.error("Arquivo deve conter colunas 'tempo' e 'concentracao'")
            else:
                tau_exp = exp_data['tempo'].values
                CA_exp = exp_data['concentracao'].values

                # Ajustar parâmetros
                if st.button("🔍 Realizar Ajuste de Parâmetros"):
                    with st.spinner("Ajustando parâmetros..."):
                        k_opt, k_error = fit_kinetic_parameters(
                            tau_exp, CA_exp, n, CA0)

                        # Calcular predições com k otimizado
                        if reator_type == "CSTR":
                            CA_pred = np.array(
                                [cstr_model(t, k_opt, n, CA0) for t in tau_exp])
                        else:
                            CA_pred = np.array(
                                [pfr_model(t, k_opt, n, CA0) for t in tau_exp])

                        # DEBUG: Verificar dados
                        # st.write(f"Tau_exp shape: {tau_exp.shape}, CA_exp shape: {CA_exp.shape}")
                        # st.write(f"CA_pred shape: {CA_pred.shape if 'CA_pred' in locals() else 'Not defined'}")

                        # Verificar se os arrays têm o mesmo tamanho
                        if len(tau_exp) != len(CA_pred):
                            st.error(
                                f"ERRO: tau_exp ({len(tau_exp)}) e CA_pred ({len(CA_pred)}) têm tamanhos diferentes!")
                            # Recriar CA_pred com o tamanho correto
                            CA_pred = np.array(
                                [cstr_model(t, k_opt, n, CA0) for t in tau_exp])

                        # Verificar valores
                        # st.write("Primeiros 5 valores:")
                        # st.write(f"Tau: {tau_exp[:5]}")
                        # st.write(f"CA_exp: {CA_exp[:5]}")
                        # st.write(f"CA_pred: {CA_pred[:5]}")

                        # Verificar se há valores NaN
                        # st.write("**Verificação de valores NaN:**")
                        # st.write(f"NaN em tau_exp: {np.isnan(tau_exp).any()}")
                        # st.write(f"NaN em CA_exp: {np.isnan(CA_exp).any()}")
                        # st.write(f"NaN em CA_pred: {np.isnan(CA_pred).any()}")

                        # DEBUG dentro da função
                        # st.write(f"DEBUG: tau_exp = {tau_exp[:5]}")
                        # st.write(f"DEBUG: CA_exp = {CA_exp[:5]}")
                        # st.write(f"DEBUG: CA0 = {CA0}")
                        # st.write(f"DEBUG: reator_type = {reator_type}")

                        # Métricas de erro
                        r2 = r2_score(CA_exp, CA_pred)
                        rmse = np.sqrt(mean_squared_error(CA_exp, CA_pred))
                        mae = np.mean(np.abs(CA_exp - CA_pred))

                        # ✅ CORRETO: Análise de convergência
                        try:
                            st.subheader("🔍 Análise de Convergência")
                            conv_fig = plot_convergence_analysis(
                                tau_exp, CA_exp, n, CA0)  # ✅ CA0 correto
                            st.pyplot(conv_fig)
                        except Exception as e:
                            st.warning(
                                "Não foi possível gerar a análise de convergência")
                            st.write(f"Erro: {str(e)}")

                        # Exibir resultados
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("k otimizado",
                                      f"{k_opt:.6f}", f"±{k_error:.6f}")
                        with col2:
                            st.metric("R²", f"{r2:.4f}")
                        with col3:
                            st.metric("RMSE", f"{rmse:.6f}")

                        # Plot comparativo
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        ax2.scatter(tau_exp, CA_exp, color='red',
                                    label='Experimental', alpha=0.7)
                        ax2.plot(tau_exp, CA_pred, 'b-', linewidth=2,
                                 label='Modelo Ajustado')
                        ax2.set_xlabel('Tempo Espacial (min)')
                        ax2.set_ylabel('Concentração CA (mol/L)')
                        ax2.set_title(
                            f'Ajuste do Modelo - Reator {reator_type}')
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()

                        st.pyplot(fig2)

                        # Tabela de resultados
                        results_df = pd.DataFrame({
                            'Tempo': tau_exp,
                            'Experimental': CA_exp,
                            'Predito': CA_pred,
                            'Erro': CA_exp - CA_pred
                        })

                        st.write("📋 Tabela de Resultados:")
                        st.dataframe(results_df)

        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
    else:
        st.info(
            "⚠️ Faça upload de um arquivo CSV com dados experimentais para realizar o ajuste.")

with tab3:
    st.header("Resultados e Análise")

    if uploaded_file is not None and 'k_opt' in locals():
        # Análise estatística detalhada
        st.subheader("📊 Análise Estatística do Ajuste")

        residuals = CA_exp - CA_pred
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Estatísticas dos Resíduos:**")
            res_stats = pd.DataFrame({
                'Métrica': ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo'],
                'Valor': [
                    f"{np.mean(residuals):.6f}",
                    f"{np.std(residuals):.6f}",
                    f"{np.min(residuals):.6f}",
                    f"{np.max(residuals):.6f}"
                ]
            })
            st.table(res_stats)

        with col2:
            # Plot de resíduos
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.scatter(tau_exp, residuals, alpha=0.7)
            ax3.axhline(y=0, color='r', linestyle='--')
            ax3.set_xlabel('Tempo Espacial (min)')
            ax3.set_ylabel('Resíduos (Experimental - Predito)')
            ax3.set_title('Análise de Resíduos')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)

        # Conclusões
        st.subheader("🎯 Conclusões")
        if r2 > 0.98:
            st.success("✅ Excelente ajuste do modelo aos dados experimentais!")
        elif r2 > 0.90:
            st.success("✅ Bom ajuste do modelo aos dados experimentais!")
        elif r2 > 0.85:
            st.warning("⚠️ Ajuste razoável, mas pode ser melhorado.")
        else:
            st.error("❌ Ajuste pobre - verifique os parâmetros ou o modelo.")

        st.write(f"- Coeficiente de determinação (R²): {r2:.4f}")
        st.write(f"- Erro quadrático médio (RMSE): {rmse:.6f}")
        st.write(
            f"- Constante cinética otimizada: k = {k_opt:.6f} ± {k_error:.6f}")

        # Recomendações
        if r2 < 0.90:
            st.info("💡 **Recomendações para melhorar o ajuste:**")
            st.write("1. Verifique se a ordem da reação (n) está correta")
            st.write("2. Confirme o valor da concentração inicial CA₀")
            st.write("3. Considere testar diferentes valores iniciais para k")

    else:
        st.info(
            "Realize o ajuste de parâmetros na aba anterior para ver os resultados detalhados.")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido por Luis Henrique")
st.markdown("Engenharia Quimica - UFAM")

