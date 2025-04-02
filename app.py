import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.title("💻 Machine Learning App")
st.markdown("### This app will predict your **obesity level**!")

@st.cache_data
def load_data():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
    return df

df = load_data()

with st.expander("🧐 Data", expanded=True):
    st.markdown("This is a raw data")
    st.dataframe(df)

with st.expander("📊 Data Visualization", expanded=True):
    st.markdown("### Data Visualization")

    fig = px.scatter(
        df,
        x="Height",
        y="Weight",
        color="NObeyesdad",  # Target label
        labels={"NObeyesdad": "Obesity Level"},
        title="Height vs Weight by Obesity Level"
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def load_artifacts():
    with open("obesity_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("preprocessing.pkl", "rb") as file:
        preprocess_encode = pickle.load(file)
    with open("label_encoder_target.pkl", "rb") as file:
        label_encoder_target = pickle.load(file)
    with open("standard_scaler.pkl", "rb") as file:
        standard_scaler = pickle.load(file)
    with open("robust_scaler.pkl", "rb") as file:
        robust_scaler = pickle.load(file)
    with open("scaler_columns.pkl", "rb") as file:
        scaler_cols = pickle.load(file)
    return model, preprocess_encode, label_encoder_target, standard_scaler, robust_scaler, scaler_cols

# Preprocess
def preprocess_user_input(user_input, preprocess_encode, scaler_cols, robust_scaler, standard_scaler):
    df = pd.DataFrame([user_input])
    
    for col in preprocess_encode["ordinal_cols"]:
        df[col] = df[col].map(preprocess_encode["ordinal_mappings"][col])
    nominal_cols = preprocess_encode["nominal_cols"]
    one_hot_encoder = preprocess_encode["one_hot_encoder"]
    
    encoded = one_hot_encoder.transform(df[nominal_cols])
    encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(), index=df.index)
    
    df.drop(columns=nominal_cols, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    
    if scaler_cols.get("robust"):
        df[robust_scaler.feature_names_in_] = robust_scaler.transform(df[robust_scaler.feature_names_in_])
    if scaler_cols.get("standard"):
        df[standard_scaler.feature_names_in_] = standard_scaler.transform(df[standard_scaler.feature_names_in_])
        
    return df


model, preprocess_encode, label_encoder, standard_scaler, robust_scaler, scaler_cols = load_artifacts()

# ============ Input UI =============
st.markdown("### Input Data by User")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 10, 100, 25)
    height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0)
    fam_history = st.selectbox("Family History With Overweight", ["yes", "no"])
    favc = st.selectbox("High Caloric Food Consumption (FAVC)", ["yes", "no"])
    fcvc = st.slider("FCVC (Veggie Consumption)", 1.0, 3.0, step=0.1)
    ncp = st.slider("NCP (Main Meals)", 1.0, 4.0, step=0.1)

with col2:
    caec = st.selectbox("CAEC (Snacks Between Meals)", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Do you smoke?", ["yes", "no"])
    ch2o = st.slider("Water Intake (CH2O)", 1.0, 3.0, step=0.1)
    scc = st.selectbox("Calories Monitor (SCC)", ["yes", "no"])
    faf = st.slider("Physical Activity (FAF)", 0.0, 3.0, step=0.1)
    tue = st.slider("Technology Usage (TUE)", 0.0, 20.0, step=0.1)
    calc = st.selectbox("Alcohol Consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportation (MTRANS)", ["Automobile", "Motorbike", "Walking", "Public_Transportation", "Bike"])

user_input = {
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "family_history_with_overweight": fam_history,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "SCC": scc,
    "FAF": faf,
    "TUE": tue,
    "CALC": calc,
    "MTRANS": mtrans
}

# ============ Predict Button =============
if st.button("Predict"):
    X_user = preprocess_user_input(user_input, preprocess_encode, scaler_cols, robust_scaler, standard_scaler)
    proba = model.predict_proba(X_user)[0]
    prediction = model.predict(X_user)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    st.subheader("📥 Data input by user")
    st.dataframe(pd.DataFrame([user_input]))

    st.subheader("📊 Obesity Prediction Probabilities")
    proba_df = pd.DataFrame([proba], columns=label_encoder.classes_)
    st.dataframe(proba_df.style.format("{:.4f}"))
    
    st.markdown("#### 📈 Probability per Class (Bar Chart)")
    fig_bar = px.bar(
        x=label_encoder.classes_,
        y=proba,
        labels={"x": "Obesity Class", "y": "Probability"},
        text=[f"{p:.2%}" for p in proba],
        color=label_encoder.classes_,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(yaxis_range=[0, 1], height=400)
    
    st.plotly_chart(fig_bar, use_container_width=True)


    st.success(f"🎯 The predicted output is: **{label}**")







    