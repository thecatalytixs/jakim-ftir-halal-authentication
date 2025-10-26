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

# Force matplotlib to use light theme
plt.style.use('default')

st.set_page_config(page_title="JAKIM FTIR Halal Authentication Platform", layout="wide")
st.title("Halal Authentication Platform")

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
pca = PCA(n_components=3)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=pca_scores, columns=["PC1", "PC2", "PC3"])
pca_df["Class"] = y.values
pca_df["SampleID"] = df["SampleID"].values

# Option to toggle SampleID labels
show_labels = st.checkbox("Show SampleID labels on PCA plot")
if show_labels:
    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Class", text="SampleID", title="PCA Score Plot (3D)")
    fig.update_traces(textposition='top center')
else:
    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Class", title="PCA Score Plot (3D)")

st.plotly_chart(fig, use_container_width=True)

# PCA Loadings (Variable Plot)
st.subheader("3. Variable Plot (PCA Loadings)")
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2", "PC3"], index=X.columns)
fig_loadings = px.scatter_3d(loadings.reset_index(), x="PC1", y="PC2", z="PC3", text="index")
fig_loadings.update_layout(title="PCA Loadings Plot (PC1 vs PC2 vs PC3)", scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
st.plotly_chart(fig_loadings, use_container_width=True)

# PCA Biplot
st.subheader("4. PCA Biplot")
fig_biplot = go.Figure()

# Add scores
for label in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == label]
    fig_biplot.add_trace(go.Scatter3d(x=subset["PC1"], y=subset["PC2"], z=subset["PC3"],
                                      mode='markers+text' if show_labels else 'markers',
                                      text=subset["SampleID"] if show_labels else None,
                                      name=label))

# Add loadings
for i in range(loadings.shape[0]):
    fig_biplot.add_trace(go.Scatter3d(x=[0, loadings.iloc[i, 0]*3], y=[0, loadings.iloc[i, 1]*3], z=[0, loadings.iloc[i, 2]*3],
                                      mode='lines+text',
                                      text=["", loadings.index[i]],
                                      name=loadings.index[i],
                                      line=dict(color='black', width=2)))

fig_biplot.update_layout(title="PCA Biplot (PC1 vs PC2 vs PC3)",
                         scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
st.plotly_chart(fig_biplot, use_container_width=True)

# PLS-DA Scores Plot
st.subheader("5. PLS-DA Observation Plot")
pls = PLSRegression(n_components=3)
pls.fit(X_scaled, y_encoded)
pls_scores = pls.x_scores_
pls_df = pd.DataFrame(pls_scores, columns=["PLS1", "PLS2", "PLS3"])
pls_df["Class"] = y.values
pls_df["SampleID"] = df["SampleID"].values

show_pls_labels = st.checkbox("Show SampleID labels on PLS-DA plot")
if show_pls_labels:
    fig_pls = px.scatter_3d(pls_df, x="PLS1", y="PLS2", z="PLS3", color="Class", text="SampleID", title="PLS-DA Observation Plot (3D)")
    fig_pls.update_traces(textposition='top center')
else:
    fig_pls = px.scatter_3d(pls_df, x="PLS1", y="PLS2", z="PLS3", color="Class", title="PLS-DA Observation Plot (3D)")

st.plotly_chart(fig_pls, use_container_width=True)

# PLS-DA Classification Report (Full Set)
y_pred_pls = pls.predict(X_scaled)
y_pred_labels = np.round(y_pred_pls).astype(int).flatten()
y_pred_labels = np.clip(y_pred_labels, 0, len(label_encoder.classes_) - 1)
y_pred_classes = label_encoder.inverse_transform(y_pred_labels)

report_dict = classification_report(y, y_pred_classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)
st.markdown("**Classification Report (Training Set via PLS-DA):**")
st.dataframe(report_df, use_container_width=True)

pls_conf_matrix = confusion_matrix(y, y_pred_classes)
st.markdown("**Confusion Matrix (Training Set via PLS-DA):**")
st.dataframe(pd.DataFrame(pls_conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

# PLS-DA VIP Scores
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

# Classification with Logistic Regression (Test Set)
st.subheader("6. Logistic Regression Classification (Test Set)")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

report_dict_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
report_df_test = pd.DataFrame(report_dict_test).transpose().round(2)
st.markdown("**Classification Report (Test Set):**")
st.dataframe(report_df_test, use_container_width=True)

conf_matrix_test = confusion_matrix(y_test, y_pred_test)
st.markdown("**Confusion Matrix (Test Set):**")
st.dataframe(pd.DataFrame(conf_matrix_test, index=label_encoder.classes_, columns=label_encoder.classes_))

