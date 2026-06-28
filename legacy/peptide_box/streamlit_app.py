"""
streamlit_app.py

Interactive visualization of θ-plane and C′–N–O subplane geometry from *_normals.csv
output of planes_from_backbone_ortho_boxes.py.

Features
--------
• Select a single θ box (plane_index)
• Hide all other boxes (focus mode)
• Show θ-plane and C′–N–O plane in different colors
• Show normals (θ up, CNO down)
• Rotate the C′–N–O normal interactively
• Observe RMS behavior conceptually

Run:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------- math helpers ----------------------------

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rotate_normal(n, axis, angle_deg):
    """Rodrigues rotation"""
    axis = normalize(axis)
    n = normalize(n)
    theta = np.deg2rad(angle_deg)
    return (
        n * np.cos(theta)
        + np.cross(axis, n) * np.sin(theta)
        + axis * np.dot(axis, n) * (1 - np.cos(theta))
    )


def plane_square(center, normal, size=3.0):
    """Generate a square patch representing a plane"""
    n = normalize(normal)
    if abs(n[0]) < 0.9:
        u = normalize(np.cross(n, [1, 0, 0]))
    else:
        u = normalize(np.cross(n, [0, 1, 0]))
    v = np.cross(n, u)

    c = np.asarray(center)
    return np.array([
        c + size * ( u + v),
        c + size * ( u - v),
        c + size * (-u - v),
        c + size * (-u + v),
    ])


# ---------------------------- streamlit UI ----------------------------

st.set_page_config(layout="wide")
st.title("θ-plane vs C′–N–O Plane Explorer")

st.markdown(
    """
This app isolates **one peptide step at a time** and visualizes the geometric
relationship between the **θ-plane** and the **C′–N–O subplane**.

The goal is to show how **twist / pucker of C′–N–O** drives the RMS measured
relative to the θ-plane.
"""
)

# ---------------------------- data input ----------------------------

csv_file = st.sidebar.file_uploader(
    "Upload *_normals.csv",
    type=["csv"]
)

if csv_file is None:
    st.info("Upload a *_normals.csv file to begin")
    st.stop()

df = pd.read_csv(csv_file)

for col in ["nx", "ny", "nz", "cno_nx", "cno_ny", "cno_nz"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------- focus selection ----------------------------

plane_index = st.sidebar.selectbox(
    "Select θ plane (plane_index)",
    df["plane_index"].tolist()
)

row = df[df.plane_index == plane_index].iloc[0]

theta_normal = np.array([row.nx, row.ny, row.nz])
cno_normal_base = np.array([row.cno_nx, row.cno_ny, row.cno_nz])

# Use origin as shared center for conceptual clarity
center = np.zeros(3)

# ---------------------------- controls ----------------------------

st.sidebar.markdown("### C′–N–O Normal Manipulation")

angle = st.sidebar.slider(
    "Rotate C′–N–O normal (degrees)",
    -180.0, 180.0, 0.0, 1.0
)

axis_choice = st.sidebar.selectbox(
    "Rotation axis",
    ["θ normal", "x-axis", "y-axis", "z-axis"]
)

if axis_choice == "θ normal":
    axis = theta_normal
elif axis_choice == "x-axis":
    axis = np.array([1, 0, 0])
elif axis_choice == "y-axis":
    axis = np.array([0, 1, 0])
else:
    axis = np.array([0, 0, 1])

cno_normal = rotate_normal(cno_normal_base, axis, angle)

# ---------------------------- geometry ----------------------------

theta_plane = plane_square(center, theta_normal, size=3.0)
cno_plane   = plane_square(center, cno_normal, size=2.0)

# ---------------------------- plot ----------------------------

fig = go.Figure()

fig.add_trace(go.Mesh3d(
    x=theta_plane[:, 0],
    y=theta_plane[:, 1],
    z=theta_plane[:, 2],
    color="royalblue",
    opacity=0.35,
    name="θ-plane"
))

fig.add_trace(go.Mesh3d(
    x=cno_plane[:, 0],
    y=cno_plane[:, 1],
    z=cno_plane[:, 2],
    color="firebrick",
    opacity=0.35,
    name="C′–N–O plane"
))

# θ normal (up)
fig.add_trace(go.Scatter3d(
    x=[0, theta_normal[0]],
    y=[0, theta_normal[1]],
    z=[0, theta_normal[2]],
    mode="lines",
    line=dict(color="royalblue", width=7),
    name="θ normal"
))

# CNO normal (down)
fig.add_trace(go.Scatter3d(
    x=[0, -cno_normal[0]],
    y=[0, -cno_normal[1]],
    z=[0, -cno_normal[2]],
    mode="lines",
    line=dict(color="firebrick", width=7),
    name="C′–N–O normal"
))

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------- readouts ----------------------------

col1, col2 = st.columns(2)
col1.metric("θ-plane RMS (from file)", f"{row.rms:.4f} Å")
col2.metric("C′–N–O → θ RMS", f"{row.cno_rms:.4f} Å")

st.markdown(
    """
### Interpretation
• When the C′–N–O normal is **antiparallel** to the θ normal, RMS → 0
• Twisting or puckering the C′–N–O plane increases RMS smoothly
• RMS is therefore a **geometric proxy** for subplane misalignment

This view intentionally removes all other peptide planes to expose the
**local geometric mechanism**.
"""
)
