import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

plt.style.use('default')

st.title("FTIR-Based Halal Authentication Platform")

uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    label_col = st.selectbox("Select the label column", df.columns)

    features = df.drop(columns=[label_col])
    labels = df[label_col]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA Analysis
    st.subheader("PCA Analysis")
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=components, columns=["PC1", "PC2"])
    pca_df["Label"] = labels.values

    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Label", title="PCA Score Plot")
    st.plotly_chart(fig_pca)

    # PCA Loadings Plot
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features.columns)
    st.subheader("PCA Variable Plot")
    fig_loadings = px.scatter(loadings, x="PC1", y="PC2", text=loadings.index, title="PCA Variable Plot")
    fig_loadings.update_traces(textposition='top center')
    st.plotly_chart(fig_loadings)

    # Biplot
    st.subheader("PCA Biplot")
    fig_biplot = px.scatter(pca_df, x="PC1", y="PC2", color="Label")
    for i in range(len(loadings)):
        fig_biplot.add_shape(type='line', x0=0, y0=0, x1=loadings.iloc[i, 0]*5, y1=loadings.iloc[i, 1]*5,
                             line=dict(color='black', width=1))
        fig_biplot.add_annotation(x=loadings.iloc[i, 0]*5, y=loadings.iloc[i, 1]*5,
                                  ax=0, ay=0, xanchor="center", yanchor="bottom",
                                  text=loadings.index[i], showarrow=True, arrowhead=2)
    st.plotly_chart(fig_biplot)

    # PLS-DA Analysis
    st.subheader("PLS-DA Analysis")
    pls = PLSRegression(n_components=2)
    pls_fit = pls.fit(X_scaled, y_encoded)
    pls_scores = pls_fit.x_scores_
    pls_df = pd.DataFrame(data=pls_scores, columns=["PLS1", "PLS2"])
    pls_df["Label"] = labels.values

    fig_plsda = px.scatter(pls_df, x="PLS1", y="PLS2", color="Label", title="PLS-DA Score Plot")
    st.plotly_chart(fig_plsda)

    # Variable Importance in Projection (VIP)
    st.subheader("Variable Importance (VIP) Scores")
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    vip_scores = pd.Series(vips, index=features.columns)
    st.dataframe(vip_scores.sort_values(ascending=False).reset_index().rename(columns={"index": "Variable", 0: "VIP Score"}))

    # Logistic Regression Classifier
    st.subheader("Classification using Logistic Regression")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    st.text("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=label_encoder.classes_, y=label_encoder.classes_,
                                       colorscale='Blues', showscale=True))
    fig_cm.update_layout(title="Confusion Matrix (Test Set)", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm)

    # Cross-validation
    st.subheader("Cross-Validation Accuracy")
    scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    st.write(f"Average Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

    # Confusion Matrix from Cross-Validation
    st.subheader("Cross-Validation Confusion Matrix")
    y_cv_pred = cross_val_predict(model, X_scaled, y_encoded, cv=5)
    cm_cv = confusion_matrix(y_encoded, y_cv_pred)
    fig_cm_cv = go.Figure(data=go.Heatmap(z=cm_cv, x=label_encoder.classes_, y=label_encoder.classes_,
                                          colorscale='Purples', showscale=True))
    fig_cm_cv.update_layout(title="Confusion Matrix (Cross-Validation)", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm_cv)
