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
