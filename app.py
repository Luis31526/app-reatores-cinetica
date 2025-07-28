import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Cálculo de Cinética & Adsorção", layout="wide")

st.title("Reatores CSTR/PFR e Adsorção (Langmuir e Freundlich)")

aba = st.sidebar.radio("Escolha a aba:", ["Cálculo Cinético", "Adsorção - Langmuir/Freundlich"])

if aba == "Cálculo Cinético":
    st.header("Estimativa da Constante Cinética")

    reator = st.selectbox("Tipo de Reator", ["CSTR", "PFR"])
    Q = st.number_input("Vazão Volumétrica (L/min)", min_value=0.0)
    V = st.number_input("Volume do Reator (L)", min_value=0.0)
    C0 = st.number_input("Concentração Inicial (mol/L)", min_value=0.0)
    Cf = st.number_input("Concentração Final (mol/L)", min_value=0.0)

    if st.button("Calcular k"):
        if reator == "CSTR":
            try:
                k = Q * (C0 - Cf) / (V * Cf)
                st.success(f"k = {k:.4f} 1/min")
            except:
                st.error("Erro no cálculo! Verifique os dados.")
        elif reator == "PFR":
            try:
                k = -(Q/V) * np.log(Cf/C0)
                st.success(f"k = {k:.4f} 1/min")
            except:
                st.error("Erro no cálculo! Verifique os dados.")

elif aba == "Adsorção - Langmuir/Freundlich":
    st.header("Modelos de Adsorção")

    modelo = st.selectbox("Modelo", ["Langmuir", "Freundlich"])
    dados = st.file_uploader("Importe um CSV com colunas: Ce (mg/L), qe (mg/g)", type="csv")

    if dados:
        df = pd.read_csv(dados)
        st.write("Dados carregados:")
        st.dataframe(df)

        Ce = df["Ce"]
        qe = df["qe"]

        if modelo == "Langmuir":
            inv_Ce = 1 / Ce
            inv_qe = 1 / qe
            coef = np.polyfit(inv_Ce, inv_qe, 1)
            a = coef[0]
            b = coef[1]
            qmax = 1 / b
            KL = a * qmax
            st.success(f"qmax = {qmax:.4f} mg/g")
            st.success(f"KL = {KL:.4f} L/mg")

        elif modelo == "Freundlich":
            log_Ce = np.log10(Ce)
            log_qe = np.log10(qe)
            coef = np.polyfit(log_Ce, log_qe, 1)
            n = 1 / coef[0]
            Kf = 10**coef[1]
            st.success(f"Kf = {Kf:.4f}")
            st.success(f"n = {n:.4f}")
