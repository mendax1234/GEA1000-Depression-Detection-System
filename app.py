import streamlit as st
import pandas as pd
import joblib

# Load saved objects
scaler = joblib.load('model/scaler.joblib')
model = joblib.load('model/random_forest_model.joblib')
categorical_options = joblib.load('model/categorical_options.joblib')
feature_names = joblib.load('model/feature_names.joblib')

# Define nominal_columns (fixing previous error)
nominal_columns = ['Gender', 'City', 'Profession', 'Dietary_Habits', 'Degree', 
                   'Have_you_ever_had_suicidal_thoughts_', 'Family_History_of_Mental_Illness', 
                   'Sleep_Duration']

# Initialize session state for view management
if 'view' not in st.session_state:
    st.session_state.view = 'input'  # Default view is the input form
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Function to display the input form
def show_input_form():
    st.title("Depression Prediction App")
    st.write("""
        This app predicts whether you might be experiencing depression based on your inputs.
        Please provide the following information and click 'Predict'.
        **Note:** This is a model prediction, not a medical diagnosis. Consult a professional for health concerns.
    """)

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = col1.slider("Age", min_value=18, max_value=80, value=25)
            gender = col1.selectbox("Gender", options=categorical_options['Gender'])
            city = col1.selectbox("City", options=categorical_options['City'])
            profession = col1.selectbox("Profession", options=categorical_options['Profession'])
            degree = col1.selectbox("Degree", options=categorical_options['Degree'])
        
        with col2:
            cgpa = col2.slider("CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
            work_study_hours = col2.slider("Work/Study Hours per Week", min_value=0, max_value=80, value=40)
            dietary_habits = col2.selectbox("Dietary Habits", options=categorical_options['Dietary_Habits'])
            sleep_duration = col2.selectbox("Sleep Duration", options=categorical_options['Sleep_Duration'])
            suicidal_thoughts = col2.radio("Suicidal Thoughts?", options=categorical_options['Have_you_ever_had_suicidal_thoughts_'])
            family_history = col2.radio("Family History?", options=categorical_options['Family_History_of_Mental_Illness'])
        
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

        # Convert to DataFrame
        user_df = pd.DataFrame([user_input])

        # Preprocess the input
        try:
            numerical_columns = ['Age', 'CGPA', 'Work_Study_Hours']
            user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])
            user_df_encoded = pd.get_dummies(user_df, columns=nominal_columns, drop_first=True)
            missing_cols = set(feature_names) - set(user_df_encoded.columns)
            for col in missing_cols:
                user_df_encoded[col] = 0
            user_df_encoded = user_df_encoded[feature_names]

            # Make prediction
            prediction = model.predict(user_df_encoded)
            st.session_state.prediction = prediction[0]
            
            # Switch view based on prediction
            if prediction[0] == 'Yes':  # Adjust based on your model's output
                st.session_state.view = 'warning'
            else:
                st.session_state.view = 'safe'

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
        </style>
        <div class="warning-box">
            <h2 style="color: red;">⚠️ Warning: Depression Tendency Detected</h2>
            <p>The model predicts you might be experiencing depression. Please consider consulting a mental health professional.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    if st.button("Back to Input Form"):
        st.session_state.view = 'input'
        st.session_state.prediction = None

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
        </style>
        <div class="safe-box">
            <h2 style="color: green;">✅ Safe: No Depression Tendency</h2>
            <p>The model predicts you are not experiencing depression. Stay well!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    if st.button("Back to Input Form"):
        st.session_state.view = 'input'
        st.session_state.prediction = None

# Main app logic based on current view
if st.session_state.view == 'input':
    show_input_form()
elif st.session_state.view == 'warning':
    show_warning()
elif st.session_state.view == 'safe':
    show_safe()