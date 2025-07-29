import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="FTIR-Based Halal Authentication", layout="wide")
st.title("FTIR-Based Halal Authentication Platform")

# File uploader
uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("simulated_ftir_data.csv")  # fallback demo file

df = load_data(uploaded_file)
st.subheader("1. Preview of Uploaded Dataset")
st.dataframe(df.head())

# Extract features and class labels
X = df.drop(columns=["SampleID", "Class"])
y = df["Class"]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# PCA
st.subheader("2. Principal Component Analysis (PCA)")
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=pca_scores, columns=["PC1", "PC2"])
pca_df["Class"] = y.values
pca_df["SampleID"] = df["SampleID"].values

# Option to toggle SampleID labels
show_labels = st.checkbox("Show SampleID labels on PCA plot")
if show_labels:
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Class", text="SampleID", title="PCA Score Plot")
    fig.update_traces(textposition='top center')
else:
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Class", title="PCA Score Plot")

st.plotly_chart(fig, use_container_width=True)

# Matplotlib PNG Export
fig_pca, ax = plt.subplots()
for label in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == label]
    ax.scatter(subset["PC1"], subset["PC2"], label=label)
ax.set_title("PCA Score Plot")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
buf = io.BytesIO()
fig_pca.savefig(buf, format="png")
st.download_button("Download PCA Plot as PNG", data=buf.getvalue(), file_name="pca_plot.png", mime="image/png")

# PCA Loadings (Variable Plot)
st.subheader("3. Variable Plot (PCA Loadings)")
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=X.columns)
fig_loadings, ax_load = plt.subplots()
ax_load.scatter(loadings["PC1"], loadings["PC2"])
for i, txt in enumerate(loadings.index):
    ax_load.annotate(txt, (loadings["PC1"][i], loadings["PC2"][i]))
ax_load.set_title("PCA Variable Plot (Loadings)")
ax_load.set_xlabel("PC1")
ax_load.set_ylabel("PC2")
buf_load = io.BytesIO()
fig_loadings.savefig(buf_load, format="png")
st.pyplot(fig_loadings)
st.download_button("Download Variable Plot as PNG", data=buf_load.getvalue(), file_name="variable_plot.png", mime="image/png")

# PCA Biplot
st.subheader("4. PCA Biplot")
fig_biplot, ax_bi = plt.subplots()
for label in pca_df["Class"].unique():
    filtered = pca_df[pca_df["Class"] == label]
    ax_bi.scatter(filtered["PC1"], filtered["PC2"], label=label)
    if show_labels:
        for i in range(len(filtered)):
            ax_bi.annotate(filtered["SampleID"].iloc[i], (filtered["PC1"].iloc[i], filtered["PC2"].iloc[i]))
for i in range(loadings.shape[0]):
    ax_bi.arrow(0, 0, loadings.iloc[i, 0]*5, loadings.iloc[i, 1]*5, color='r', alpha=0.5)
    ax_bi.text(loadings.iloc[i, 0]*5, loadings.iloc[i, 1]*5, loadings.index[i], color='r')
ax_bi.set_title("PCA Biplot")
ax_bi.set_xlabel("PC1")
ax_bi.set_ylabel("PC2")
ax_bi.legend()
buf_biplot = io.BytesIO()
fig_biplot.savefig(buf_biplot, format="png")
st.pyplot(fig_biplot)
st.download_button("Download PCA Biplot as PNG", data=buf_biplot.getvalue(), file_name="pca_biplot.png", mime="image/png")

# Classification Model
st.subheader("5. Halal vs Haram Classification")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.markdown("**Classification Report (Test Set):**")
report_dict_test = classification_report(y_test, y_pred, output_dict=True)
report_df_test = pd.DataFrame(report_dict_test).transpose().round(2)
st.dataframe(report_df_test.style.set_properties(**{'text-align': 'left'}), use_container_width=True)
st.download_button("Download Test Classification Report as CSV", report_df_test.to_csv().encode(), file_name="test_classification_report.csv")

conf_matrix = confusion_matrix(y_test, y_pred)
st.markdown("**Confusion Matrix (Test Set):**")
conf_df = pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_)
st.dataframe(conf_df)
st.download_button("Download Test Confusion Matrix as CSV", conf_df.to_csv().encode(), file_name="test_confusion_matrix.csv")

# PLS-DA and VIP Scores
st.subheader("6. PLS-DA and VIP Scores")
pls = PLSRegression(n_components=2)
pls.fit(X_scaled, y_encoded)
y_pred_pls = pls.predict(X_scaled)

# Fix for classification output
y_pred_labels = np.round(y_pred_pls).astype(int).flatten()
y_pred_labels = np.clip(y_pred_labels, 0, len(label_encoder.classes_) - 1)
y_pred_pls_class = label_encoder.inverse_transform(y_pred_labels)

# Classification report as a formatted table
report_dict = classification_report(y, y_pred_pls_class, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)
st.markdown("**Classification Report (Training Set via PLS-DA):**")
st.dataframe(report_df.style.set_properties(**{'text-align': 'left'}), use_container_width=True)
st.download_button("Download Training Classification Report as CSV", report_df.to_csv().encode(), file_name="training_classification_report.csv")

pls_conf_matrix = confusion_matrix(y, y_pred_pls_class)
st.markdown("**Confusion Matrix (Training Set via PLS-DA):**")
pls_conf_df = pd.DataFrame(pls_conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
st.dataframe(pls_conf_df)
st.download_button("Download Training Confusion Matrix as CSV", pls_conf_df.to_csv().encode(), file_name="training_confusion_matrix.csv")

# Observation plot for PLS-DA
st.subheader("7. PLS-DA Observation Plot")
pls_scores = pls.x_scores_
pls_df = pd.DataFrame(pls_scores, columns=["PLS1", "PLS2"])
pls_df["Class"] = y.values
pls_df["SampleID"] = df["SampleID"].values

show_labels_plsda = st.checkbox("Show SampleID labels on PLS-DA plot")
fig_plsda, ax2 = plt.subplots()
for label in pls_df["Class"].unique():
    subset = pls_df[pls_df["Class"] == label]
    ax2.scatter(subset["PLS1"], subset["PLS2"], label=label)
    if show_labels_plsda:
        for i in range(len(subset)):
            ax2.annotate(subset["SampleID"].iloc[i], (subset["PLS1"].iloc[i], subset["PLS2"].iloc[i]))
ax2.set_title("PLS-DA Observation Plot")
ax2.set_xlabel("PLS1")
ax2.set_ylabel("PLS2")
ax2.legend()
st.pyplot(fig_plsda)
buf2 = io.BytesIO()
fig_plsda.savefig(buf2, format="png")
st.download_button("Download PLS-DA Plot as PNG", data=buf2.getvalue(), file_name="plsda_observation_plot.png", mime="image/png")

# Calculate VIP scores (corrected)
T = pls.x_scores_
W = pls.x_weights_
Q = pls.y_loadings_
p, h = W.shape
SStotal = np.sum(np.square(T), axis=0) * np.square(Q).flatten()
vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))

vip_df = pd.DataFrame({'Variable': X.columns, 'VIP_Score': vip})
vip_df = vip_df.sort_values(by='VIP_Score', ascending=False)

# VIP bar chart using matplotlib
fig_vip, ax_vip = plt.subplots(figsize=(10, 6))
ax_vip.bar(vip_df['Variable'][:20], vip_df['VIP_Score'][:20])
ax_vip.set_xticklabels(vip_df['Variable'][:20], rotation=45, ha='right')
ax_vip.set_title('Top 20 VIP Scores')
ax_vip.set_ylabel('VIP Score')
buf_vip = io.BytesIO()
fig_vip.tight_layout()
fig_vip.savefig(buf_vip, format="png")
st.pyplot(fig_vip)
st.download_button("Download VIP Score Plot as PNG", data=buf_vip.getvalue(), file_name="vip_scores.png", mime="image/png")
st.download_button("Download VIP Score Data as CSV", vip_df.to_csv(index=False).encode(), file_name="vip_scores.csv")
st.dataframe(vip_df)

st.success("All charts now include PNG download functionality.")
