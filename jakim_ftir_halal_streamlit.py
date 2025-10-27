import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import os

plt.style.use('default')

st.set_page_config(page_title="JAKIM FTIR Halal Authentication Platform", layout="wide")
st.title("JAKIM FTIR Halal Authentication Platform")
st.caption("Upload a CSV with columns SampleID Class and spectral variables such as 4000..400 cm⁻¹")

uploaded_file = st.file_uploader("Upload your FTIR dataset CSV only", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    demo_path = "simulated_ftir_data.csv"
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path)
    rng = np.random.default_rng(42)
    n_per_class = 40
    wnums = np.arange(4000, 650, -10)
    p = len(wnums)

    def make_block(mu_shift):
        base = rng.normal(0, 1, size=(n_per_class, p))
        bumps_idx = rng.choice(np.arange(50, p-50), size=4, replace=False)
        for b in bumps_idx:
            base[:, b-3:b+3] += mu_shift
        return base

    X_halal = make_block(1.2)
    X_nonhalal = make_block(-1.0)
    X_mix = make_block(0.4)

    X = np.vstack([X_halal, X_nonhalal, X_mix])
    classes = (["Bovine"] * n_per_class) + (["Porcine"] * n_per_class) + (["Suspect"] * n_per_class)
    sample_ids = [f"S{i+1:03d}" for i in range(len(X))]

    df_demo = pd.DataFrame(X, columns=[f"{w}" for w in wnums])
    df_demo.insert(0, "SampleID", sample_ids)
    df_demo.insert(1, "Class", classes)
    return df_demo

df = load_data(uploaded_file)

st.subheader("1. Preview of Uploaded Dataset")
st.dataframe(df.head(), use_container_width=True)

required_cols = {"SampleID", "Class"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must include SampleID and Class columns")
    st.stop()

feature_cols = [c for c in df.columns if c not in ["SampleID", "Class"]]
if len(feature_cols) < 3:
    st.error("Not enough spectral variables found")
    st.stop()

X = df[feature_cols].copy()
y = df["Class"].copy()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

st.subheader("2. Principal Component Analysis PCA")
n_pc = st.slider("Number of PCs", min_value=2, max_value=10, value=3, step=1)

scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X)

pca = PCA(n_components=n_pc, random_state=42)
pca_scores_all = pca.fit_transform(X_all_scaled)

pca_cols = [f"PC{i+1}" for i in range(n_pc)]
pca_df = pd.DataFrame(pca_scores_all, columns=pca_cols)
pca_df["Class"] = label_encoder.inverse_transform(y_encoded)
pca_df["SampleID"] = df["SampleID"].values

show_labels = st.checkbox("Show SampleID labels on PCA plot", value=False)
if n_pc >= 3:
    fig = px.scatter_3d(
        pca_df,
        x=pca_cols[0], y=pca_cols[1], z=pca_cols[2],
        color="Class",
        text="SampleID" if show_labels else None,
        title="PCA Score Plot 3D"
    )
    if show_labels:
        fig.update_traces(textposition='top center')
else:
    fig = px.scatter(
        pca_df,
        x=pca_cols[0], y=pca_cols[1],
        color="Class",
        text="SampleID" if show_labels else None,
        title="PCA Score Plot 2D"
    )
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Variable Plot PCA Loadings")
loadings = pd.DataFrame(pca.components_.T, columns=pca_cols, index=feature_cols)
if n_pc >= 3:
    fig_loadings = px.scatter_3d(
        loadings.reset_index(), x=pca_cols[0], y=pca_cols[1], z=pca_cols[2],
        text="index", title="PCA Loadings Plot"
    )
else:
    fig_loadings = px.scatter(
        loadings.reset_index(), x=pca_cols[0], y=pca_cols[1],
        text="index", title="PCA Loadings Plot"
    )
st.plotly_chart(fig_loadings, use_container_width=True)

st.subheader("4. PCA Biplot")
fig_biplot = go.Figure()
for label in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == label]
    mode = 'markers+text' if show_labels else 'markers'
    if n_pc >= 3:
        fig_biplot.add_trace(go.Scatter3d(
            x=subset[pca_cols[0]], y=subset[pca_cols[1]], z=subset[pca_cols[2]],
            mode=mode,
            text=subset["SampleID"] if show_labels else None,
            name=label
        ))
    else:
        fig_biplot.add_trace(go.Scatter(
            x=subset[pca_cols[0]], y=subset[pca_cols[1]],
            mode=mode,
            text=subset["SampleID"] if show_labels else None,
            name=label
        ))

scale = 3.0
for i in range(loadings.shape[0]):
    if n_pc >= 3:
        fig_biplot.add_trace(go.Scatter3d(
            x=[0, loadings.iloc[i, 0]*scale],
            y=[0, loadings.iloc[i, 1]*scale],
            z=[0, loadings.iloc[i, 2]*scale],
            mode='lines+text',
            text=["", loadings.index[i]],
            name=loadings.index[i],
            line=dict(width=2)
        ))
    else:
        fig_biplot.add_trace(go.Scatter(
            x=[0, loadings.iloc[i, 0]*scale],
            y=[0, loadings.iloc[i, 1]*scale],
            mode='lines+text',
            text=["", loadings.index[i]],
            name=loadings.index[i],
            line=dict(width=2)
        ))

fig_biplot.update_layout(
    title="PCA Biplot",
    scene=dict(
        xaxis_title=pca_cols[0],
        yaxis_title=pca_cols[1],
        zaxis_title=pca_cols[2] if n_pc >= 3 else None
    )
)
st.plotly_chart(fig_biplot, use_container_width=True)

st.subheader("5. PLS scores plot for visual inspection")
n_pls_vis = st.slider("Number of PLS components for plotting", min_value=2, max_value=10, value=3, step=1)

pls_vis = PLSRegression(n_components=n_pls_vis)
pls_vis.fit(X_all_scaled, y_encoded)
pls_scores_all = pls_vis.transform(X_all_scaled)
pls_cols = [f"PLS{i+1}" for i in range(n_pls_vis)]
pls_df = pd.DataFrame(pls_scores_all, columns=pls_cols)
pls_df["Class"] = label_encoder.inverse_transform(y_encoded)
pls_df["SampleID"] = df["SampleID"].values

show_pls_labels = st.checkbox("Show SampleID labels on PLS plot", value=False)
if n_pls_vis >= 3:
    fig_pls = px.scatter_3d(
        pls_df, x=pls_cols[0], y=pls_cols[1], z=pls_cols[2],
        color="Class", text="SampleID" if show_pls_labels else None,
        title="PLS Observation Plot scores"
    )
else:
    fig_pls = px.scatter(
        pls_df, x=pls_cols[0], y=pls_cols[1],
        color="Class", text="SampleID" if show_pls_labels else None,
        title="PLS Observation Plot scores"
    )
st.plotly_chart(fig_pls, use_container_width=True)

st.subheader("6. PLS DA classification with Leave One Out Cross Validation")
n_pls = st.slider("Number of PLS components for classification", min_value=2, max_value=10, value=3, step=1)

loo = LeaveOneOut()
y_true, y_pred = [], []
pred_rows = []

X_np = X.values
y_np = y_encoded.copy()

for train_idx, test_idx in loo.split(X_np):
    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y_np[train_idx], y_np[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pls = PLSRegression(n_components=n_pls)
    pls.fit(X_train_scaled, y_train)

    X_train_scores = pls.transform(X_train_scaled)
    X_test_scores = pls.transform(X_test_scaled)

    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X_train_scores, y_train)

    y_hat = clf.predict(X_test_scores)[0]
    y_true.append(y_test[0])
    y_pred.append(y_hat)

    pred_rows.append({
        "SampleID": df.iloc[test_idx[0]]["SampleID"],
        "True": label_encoder.inverse_transform([y_test[0]])[0],
        "Pred": label_encoder.inverse_transform([y_hat])[0]
    })

y_true_labels = label_encoder.inverse_transform(y_true)
y_pred_labels = label_encoder.inverse_transform(y_pred)

acc_loo = accuracy_score(y_true_labels, y_pred_labels)
st.markdown(f"**LOOCV PLS DA Accuracy:** {acc_loo:.3f}")

report_dict_loo = classification_report(y_true_labels, y_pred_labels, output_dict=True)
report_df_loo = pd.DataFrame(report_dict_loo).transpose().round(3)
st.markdown("**Classification Report LOOCV:**")
st.dataframe(report_df_loo, use_container_width=True)

conf_loo = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
st.markdown("**Confusion Matrix LOOCV:**")
st.dataframe(pd.DataFrame(conf_loo, index=label_encoder.classes_, columns=label_encoder.classes_))

pred_df = pd.DataFrame(pred_rows)
st.markdown("**Per sample LOOCV predictions:**")
st.dataframe(pred_df, use_container_width=True)
st.download_button(
    "Download LOOCV predictions CSV",
    pred_df.to_csv(index=False).encode("utf-8"),
    file_name="loocv_predictions.csv",
    mime="text/csv"
)

st.subheader("7. VIP Scores from PLS on all data for exploration")
pls_vip = PLSRegression(n_components=min(n_pls_vis, len(feature_cols)))
pls_vip.fit(X_all_scaled, y_encoded)

T = pls_vip.x_scores_
W = pls_vip.x_weights_
Q = pls_vip.y_loadings_
p, h = W.shape
SStotal = np.sum(T**2, axis=0) * np.sum(Q**2, axis=1)
vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))

vip_df = pd.DataFrame({'Variable': feature_cols, 'VIP_Score': vip}).sort_values(by='VIP_Score', ascending=False)
fig_vip = px.bar(vip_df.head(20), x='Variable', y='VIP_Score', title='Top 20 VIP Scores exploratory')
st.plotly_chart(fig_vip, use_container_width=True)
st.dataframe(vip_df, use_container_width=True)
st.download_button("Download VIP scores CSV", vip_df.to_csv(index=False).encode('utf-8'),
                   file_name="vip_scores.csv", mime="text/csv")

with st.expander("Notes and practice guidance"):
    st.write(
        """
        PCA and the PLS scores plot above are visual only and are fitted on all samples
        LOOCV classification scales and fits inside each fold to avoid leakage
        VIP shown here is exploratory since it is fitted on all samples
        """
    )
