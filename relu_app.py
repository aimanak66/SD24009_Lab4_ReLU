import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function", page_icon="📈", layout="centered")
st.title("ReLU Activation Function")
st.write("ReLU (Rectified Linear Unit) returns 0 for negative inputs and returns the input itself for positive inputs.")
st.latex(r"f(x) = \\max(0, x)")

x_min = st.slider("Minimum input value", -20, 0, -10)
x_max = st.slider("Maximum input value", 1, 20, 10)
if x_min >= x_max:
    st.error("The maximum input must be greater than the minimum input.")
else:
    x = np.linspace(x_min, x_max, 400)
    y = np.maximum(0, x)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="ReLU")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output f(x)")
    ax.set_title("ReLU: f(x) = max(0, x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    value = st.number_input("Try an input value", value=-2.0, step=0.5)
    st.metric("ReLU output", f"{max(0.0, value):.4f}")
    st.caption("Interpretation: negative values become zero; positive values pass through unchanged.")
