import streamlit as st
import pandas as pd

st.title("DME/CDFA Descent Planner Tool")

st.markdown("### Inputs")
elevation = st.number_input("Threshold Elevation (ft)", value=50)
mda = st.number_input("MDA (Minimum Descent Altitude)", value=400)
gp_angle = st.number_input("Glide Path Angle (degrees)", value=3.0)
distance = st.number_input("Distance FAF to MAPt (NM)", value=5.0)

st.markdown("### Outputs")
rod_ftmin = round(101 * gp_angle)  # Approx ROD for 140 kt
st.write(f"Rate of Descent for 140kt: {rod_ftmin} ft/min")
