import streamlit as st
import pandas as pd
import joblib

# Load saved objects
scaler = joblib.load('model/scaler.joblib')
model = joblib.load('model/random_forest_model.joblib')
categorical_options = joblib.load('model/categorical_options.joblib')
feature_names = joblib.load('model/feature_names.joblib')

# Define nominal_columns
nominal_columns = ['Gender', 'City', 'Profession', 'Dietary_Habits', 'Degree', 
                   'Have_you_ever_had_suicidal_thoughts_', 'Family_History_of_Mental_Illness', 
                   'Sleep_Duration']

# Initialize session state
if 'view' not in st.session_state:
    st.session_state.view = 'input'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Function to display the input form
def show_input_form():
    st.title("Depression Prediction App")

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        
        # Column 1: Personal and Education Inputs
        with col1:
            age = col1.slider("Age", min_value=18, max_value=80, value=25)
            gender = col1.selectbox("Gender", options=categorical_options['Gender'])
            city = col1.selectbox("City", options=categorical_options['City'])
            profession = col1.selectbox("Profession", options=categorical_options['Profession'])
            degree = col1.selectbox("Degree", options=categorical_options['Degree'])
        
        # Column 2: Lifestyle and Mental Health Inputs
        with col2:
            cgpa = col2.slider("CGPA (0–10 Scale)", min_value=0.0, max_value=10.0, value=3.0, step=0.01)
            work_study_hours = col2.slider("Work/Study Hours per Week", min_value=0, max_value=80, value=40)
            dietary_habits = col2.selectbox("Dietary Habits", options=categorical_options['Dietary_Habits'])
            sleep_duration = col2.selectbox("Sleep Duration", options=categorical_options['Sleep_Duration'])
            suicidal_thoughts = col2.radio("Have you ever had suicidal thoughts?", options=categorical_options['Have_you_ever_had_suicidal_thoughts_'])
            family_history = col2.radio("Family History of Mental Illness?", options=categorical_options['Family_History_of_Mental_Illness'])
        
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        user_input = {
            'Age': age,
            'Gender': gender,
            'City': city,
            'Profession': profession,
            'Degree': degree,
            'CGPA': cgpa,
            'Work_Study_Hours': work_study_hours,
            'Dietary_Habits': dietary_habits,
            'Sleep_Duration': sleep_duration,
            'Have_you_ever_had_suicidal_thoughts_': suicidal_thoughts,
            'Family_History_of_Mental_Illness': family_history
        }

        user_df = pd.DataFrame([user_input])

        try:
            numerical_columns = ['Age', 'CGPA', 'Work_Study_Hours']
            user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])
            user_df_encoded = pd.get_dummies(user_df, columns=nominal_columns, drop_first=True)
            missing_cols = set(feature_names) - set(user_df_encoded.columns)
            for col in missing_cols:
                user_df_encoded[col] = 0
            user_df_encoded = user_df_encoded[feature_names]

            prediction = model.predict(user_df_encoded)
            st.session_state.prediction = prediction[0]
            
            if prediction[0] == 'Yes':
                st.session_state.view = 'warning'
            else:
                st.session_state.view = 'safe'
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to display the red warning interface
def show_warning():
    st.markdown(
        """
        <style>
        .warning-box {
            background-color: #ffcccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .warning-text {
            color: #000000;
        }
        .back-button {
            margin-top: 20px;
        }
        </style>
        <div class="warning-box">
            <h2 style="color: red;">⚠️ Warning: Depression Tendency Detected</h2>
            <p class="warning-text">The model predicts you might be experiencing depression. Please consider consulting a mental health professional.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('<div class="back-button"></div>', unsafe_allow_html=True)
    if st.button("Back to Input Form"):
        st.session_state.view = 'input'
        st.session_state.prediction = None
        st.rerun()

# Function to display the green safe interface
def show_safe():
    st.markdown(
        """
        <style>
        .safe-box {
            background-color: #ccffcc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .safe-text {
            color: #000000;
        }
        .back-button {
            margin-top: 20px;
        }
        </style>
        <div class="safe-box">
            <h2 style="color: green;">✅ Safe: No Depression Tendency</h2>
            <p class="safe-text">The model predicts you are not experiencing depression. Stay well!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('<div class="back-button"></div>', unsafe_allow_html=True)
    if st.button("Back to Input Form"):
        st.session_state.view = 'input'
        st.session_state.prediction = None
        st.rerun()

# Function to display the copyright notice
def show_copyright():
    st.markdown(
        """
        <style>
        .copyright {
            text-align: center;
            color: #666666;
            margin-top: 50px;
            font-size: 14px;
        }
        </style>
        <div class="copyright">
            © 2025 Depression Prediction App. All rights reserved.
        </div>
        """, 
        unsafe_allow_html=True
    )

# Main app logic
if st.session_state.view == 'input':
    show_input_form()
elif st.session_state.view == 'warning':
    show_warning()
elif st.session_state.view == 'safe':
    show_safe()

# Display copyright notice on all views
show_copyright()