import streamlit as st # type: ignore
import numpy as np
import joblib

# تحميل النموذج
model = joblib.load('treatment_response_model.pkl')

st.title("🔬 Breast Cancer Treatment Response Probability Prediction")

# واجهة المستخدم
age = st.number_input("📅Age ", min_value=18, max_value=100, value=40)

sex = st.selectbox("👤 Gender", ['Male', 'Female'])

tumor_type = st.selectbox("🔬 Tumor Type", ['Ductal', 'Lobular', 'Other'])

diff_grade = st.selectbox("📊 Differentiation Grade", ['Low', 'Intermediate', 'High'])

t_stage = st.selectbox("📏 T Stage", ['T1', 'T2', 'T3', 'T4'])

n_stage = st.selectbox("🧠 N Stage", ['N0', 'N1', 'N2', 'N3'])

m_stage = st.selectbox("🦠 M Stage", ['M0', 'M1'])

overall_stage = st.selectbox("📈 Overall Stage", ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])

weight_loss = st.slider("⚖️ Weight Loss Percentage", min_value=0.0, max_value=100.0, value=10.0)

tumor_location = st.selectbox("📍 Tumor Location", ['Left', 'Right', 'Central'])

smoking = st.selectbox("🚬 Smoking Status", ['Yes', 'No'])

# تحويل القيم النصية لأرقام (Encoding بسيط)
sex_map = {'Male': 0, 'Female': 1}
tumor_type_map = {'Ductal': 0, 'Lobular': 1, 'Other': 2}
diff_map = {'Low': 0, 'Intermediate': 1, 'High': 2}
stage_map = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
n_map = {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3}
m_map = {'M0': 0, 'M1': 1}
overall_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
location_map = {'Left': 0, 'Right': 1, 'Central': 2}
smoking_map = {'No': 0, 'Yes': 1}

# تجهيز البيانات للتنبؤ
input_data = np.array([[
    age,
    sex_map[sex],
    tumor_type_map[tumor_type],
    diff_map[diff_grade],
    stage_map[t_stage],
    n_map[n_stage],
    m_map[m_stage],
    overall_map[overall_stage],
    weight_loss,
    location_map[tumor_location],
    smoking_map[smoking]
]])

# عمل التنبؤ
if st.button("🔍 Predicted Response Probability"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅  Predicted Treatment Response Probability: {prediction * 100:.2f}%")
