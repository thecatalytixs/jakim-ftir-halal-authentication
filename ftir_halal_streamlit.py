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

# PCA Loadings (Variable Plot)
st.subheader("3. Variable Plot (PCA Loadings)")
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=X.columns)
fig_loadings = go.Figure()
fig_loadings.add_trace(go.Scatter(x=loadings["PC1"], y=loadings["PC2"], mode='markers+text',
text=loadings.index, textposition="top center"))
fig_loadings.update_layout(title="PCA Variable Plot (Loadings)", xaxis_title="PC1", yaxis_title="PC2")
st.plotly_chart(fig_loadings, use_container_width=True)

# PCA Biplot
st.subheader("4. PCA Biplot")
fig_biplot = go.Figure()

# Add score plot
for label in pca_df["Class"].unique():
filtered = pca_df[pca_df["Class"] == label]
fig_biplot.add_trace(go.Scatter(x=filtered["PC1"], y=filtered["PC2"], mode='markers+text' if show_labels else 'markers',
name=label, text=filtered["SampleID"] if show_labels else None,
textposition="top center" if show_labels else None))

# Add loading vectors
for i in range(loadings.shape[0]):
fig_biplot.add_trace(go.Scatter(x=[0, loadings.iloc[i, 0]*5], y=[0, loadings.iloc[i, 1]*5],
mode='lines+text', text=["", loadings.index[i]],
textposition="top center", name=loadings.index[i],
line=dict(color='black', width=1)))

fig_biplot.update_layout(title="PCA Biplot", xaxis_title="PC1", yaxis_title="PC2")
st.plotly_chart(fig_biplot, use_container_width=True)

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

conf_matrix = confusion_matrix(y_test, y_pred)
st.markdown("**Confusion Matrix (Test Set):**")
st.dataframe(pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_))

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

pls_conf_matrix = confusion_matrix(y, y_pred_pls_class)
st.markdown("**Confusion Matrix (Training Set via PLS-DA):**")
st.dataframe(pd.DataFrame(pls_conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

# Observation plot for PLS-DA
st.subheader("7. PLS-DA Observation Plot")
pls_scores = pls.x_scores_
pls_df = pd.DataFrame(pls_scores, columns=["PLS1", "PLS2"])
pls_df["Class"] = y.values
pls_df["SampleID"] = df["SampleID"].values

fig_pls_obs = px.scatter(pls_df, x="PLS1", y="PLS2", color="Class", text="SampleID" if show_labels else None,
show_labels_plsda = st.checkbox("Show SampleID labels on PLS-DA plot")
fig_pls_obs = px.scatter(pls_df, x="PLS1", y="PLS2", color="Class", text="SampleID" if show_labels_plsda else None,
title="PLS-DA Observation Plot")
fig_pls_obs.update_traces(textposition='top center' if show_labels else None)
fig_pls_obs.update_traces(textposition='top center' if show_labels_plsda else None)
st.plotly_chart(fig_pls_obs, use_container_width=True)

# Calculate VIP scores (corrected)
T = pls.x_scores_
W = pls.x_weights_
Q = pls.y_loadings_
p, h = W.shape
SStotal = np.sum(np.square(T), axis=0) * np.square(Q).flatten()
vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))

vip_df = pd.DataFrame({'Variable': X.columns, 'VIP_Score': vip})
vip_df = vip_df.sort_values(by='VIP_Score', ascending=False)

fig_vip = px.bar(vip_df.head(20), x='Variable', y='VIP_Score', title='Top 20 VIP Scores')
st.plotly_chart(fig_vip, use_container_width=True)
st.dataframe(vip_df)

st.success("PLS-DA classification matrix, observation plot, and VIP score module updated successfully.")
