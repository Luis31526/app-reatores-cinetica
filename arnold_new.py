import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Configuração da página
st.set_page_config(
    page_title="Simulador de Difusão – Célula de Arnold",
    page_icon=":test_tube:",
    # layout="wide"
)

st.title("🔬 Simulador de Difusão com Célula de Arnold")
st.markdown("""
Este simulador permite comparar dados experimentais obtidos na célula de Arnold
com modelos teóricos do coeficiente de difusão (D<sub>AB</sub>),
calculando erros e gerando visualizações.
""", unsafe_allow_html=True)

# --- Funções auxiliares ---


def Psat_antoine_acetona(TK):
    """ Pressão de vapor da acetona [Pa] via Antoine (coef válidos ~1-95 ºC) """
    TC = TK - 273.15
    A, B, C = 7.11714, 1210.595, 229.664
    Psat_mmHg = 10**(A - B/(TC + C))
    return Psat_mmHg * 133.322  # Pa


def calcular_D_AB_teorico(T, D_AB_ref, T_ref=298.0):
    return D_AB_ref * (T/T_ref)**1.75


def analisar_dataset(df_exp, T, M_A_kgmol, rho_A, P, D_AB_ref, label):
    """Processa dados de 1 CSV e retorna resultados"""
    z0 = df_exp['altura'].iloc[0]
    df_exp['delta_z2'] = df_exp['altura']**2 - z0**2

    # Regressão linear
    x = df_exp['tempo'].values
    y = df_exp['delta_z2'].values
    slope, intercept, r_value, _, _ = linregress(x, y)

    # Cálculo D_AB experimental
    P_total = P * 101325.0
    P_A1 = Psat_antoine_acetona(T)
    P_A2 = 0.0
    log_term = np.log((P_total - P_A2) / (P_total - P_A1))
    R = 8.314
    D_AB_exp = ((slope*1e-4) * rho_A * R * T) / \
        (2 * M_A_kgmol * P_total * log_term)

    # Cálculo D_AB Teórico
    D_AB_teo = calcular_D_AB_teorico(T, D_AB_ref)
    erro_rel = abs(D_AB_teo - D_AB_exp)/D_AB_teo * 100

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, 'o', label='Experimental')
    ax.plot(x, intercept + slope*x, 'r--',
            label=f"Ajuste: y={slope:.2e}x+{intercept:.2e}\nR²={r_value**2:.4f}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Z² - Z₀² (cm²)")
    ax.set_title(f"Ajuste Linear ({label})")
    ax.grid(True)
    ax.legend()

    return {
        "label": label,
        "slope": slope,
        "intercept": intercept,
        "R2": r_value**2,
        "D_AB_exp": D_AB_exp,
        "D_AB_teo": D_AB_teo,
        "erro_rel": erro_rel,
        "fig": fig
    }


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Parâmetros Teóricos")

    T_teo = st.number_input(
        "Temperatura para cálculo teórico (K):", value=298.15, step=0.1)

    sistema = st.selectbox("Sistema químico:", [
                           "Acetona-Ar", "Metanol-Ar", "Personalizado"])
    if sistema == "Personalizado":
        M_A = st.number_input("Massa molar do soluto (Kg/mol):", value=1.0)
        rho_A = st.number_input("Densidade do soluto (kg/m³):", value=1.0)
        pv = st.number_input("Pressão de vapor (Pa):", value=1.0)
    elif sistema == "Metanol-Ar":
        M_A, rho_A, pv = 0.03204, 769.9, 16915.704
        st.info("Sistema: Metanol-Ar")
        st.info("Densidade (Kg/m3): 769.9")
        st.info("Pressão de Vapor (Pa): 16915.704")
        st.info("Massa Molar (Kg/ mol): 0.03204")
    else:
        M_A, rho_A, pv = 58.08, 791.0, 24700
        st.info("Sistema: Acetona-Ar")
        st.info("Densidade (Kg/m3): 791.0")
        st.info("Pressão de Vapor (Pa): 24700")
        st.info("Massa Molar (Kg/ mol): 58.08")

    M_A_kgmol = M_A
    P = st.number_input("Pressão total (atm):", value=1.0, step=0.1)
    if sistema == "Metanol-Ar":
        D_AB_ref = st.number_input(
            "DAB de referência (×10⁻⁵, a 298K):", value=0.162, step=0.01)*1e-5
    elif sistema == "Acetona-Ar":
        D_AB_ref = st.number_input(
            "DAB de referência (×10⁻⁵, a 298K):", value=1.2, step=0.01)*1e-5
    else:
        D_AB_ref = st.number_input(
            "DAB de referência (×10⁻⁵, a 298K):", value=1.0, step=0.01)*1e-5

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(
    ["📤 Importar Dados", "📊 Ajuste Linear", "📈 Resultados"])

# Tab1 - Upload
with tab1:
    st.header("📤 Importar dois conjuntos de dados")
    uploaded_files = st.file_uploader(
        "Carregue **dois arquivos CSV** (colunas: 'tempo', 'altura')",
        type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) != 2:
            st.error(
                "⚠️ Por favor, carregue exatamente dois arquivos (duas temperaturas).")
        else:
            datasets = []
            for i, file in enumerate(uploaded_files, start=1):
                df = pd.read_csv(file)
                if not {'tempo', 'altura'}.issubset(df.columns):
                    st.error(
                        f"O arquivo {file.name} não tem colunas corretas.")
                else:
                    unidade = st.radio(f"Unidade de 'altura' no arquivo {file.name}:", [
                                       "cm", "m"], horizontal=True, key=f"u{i}")
                    if unidade == "m":
                        df['altura'] = df['altura']/100
                    T_val = st.number_input(
                        f"Temperatura do arquivo {file.name} (K):", value=318.0+5*i, key=f"T{i}")
                    st.dataframe(df.head())
                    datasets.append((df, T_val, f"Arquivo {i} - {file.name}"))
            st.session_state.datasets = datasets

# Tab2 - Ajuste Linear
with tab2:
    if "datasets" in st.session_state:
        resultados = []
        for df, T_val, label in st.session_state.datasets:
            res = analisar_dataset(df, T_val, M_A_kgmol,
                                   rho_A, P, D_AB_ref, label)
            st.pyplot(res["fig"])
            resultados.append(res)
        st.session_state.resultados = resultados

# Tab3 - Resultados
with tab3:
    if "resultados" in st.session_state:
        for res in st.session_state.resultados:
            st.subheader(f"📌 {res['label']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("D_AB Experimental", f"{res['D_AB_exp']:.2e}")
            col2.metric("D_AB Teórico", f"{res['D_AB_teo']:.2e}")
            col3.metric("Erro Relativo", f"{res['erro_rel']:.1f}%")
            st.markdown(
                f"- Equação: y = {res['slope']:.2e}x + {res['intercept']:.2e}")
            st.markdown(f"- R² = {res['R2']:.4f}")

        # Comparação final
        res1, res2 = st.session_state.resultados
        melhor = res1 if abs(res1['D_AB_exp']-res1['D_AB_teo']
                             ) < abs(res2['D_AB_exp']-res2['D_AB_teo']) else res2
        st.markdown("---")
        st.success(
            f"✅ O resultado de **{melhor['label']}** foi mais próximo do valor teórico.")
