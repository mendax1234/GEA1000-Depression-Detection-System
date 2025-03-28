import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

# Load saved objects
model = joblib.load('model/random_forest_model.joblib')
categorical_options = joblib.load('model/categorical_options.joblib')
feature_names = joblib.load('model/feature_names.joblib')

# Define column types
numerical_columns = ['Work_Study_Hours']
ordinal_columns = ['Academic_Pressure', 'Study_Satisfaction', 'Financial_Stress', 'Sleep_Duration', 'Dietary_Habits']
binary_columns = ['Have_you_ever_had_suicidal_thoughts_', 'Family_History_of_Mental_Illness']
relevant_columns = numerical_columns + ordinal_columns + binary_columns

# Define mappings
sleep_duration_mapping = {'Less than 5 hours': 0, '5-6 hours': 1, '6-7 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': -1}
dietary_habits_mapping = {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2, 'Others': -1}
binary_mapping = {'No': 0, 'Yes': 1}

# Initialize session state
if 'view' not in st.session_state:
    st.session_state.view = 'input'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'mode' not in st.session_state:
    st.session_state.mode = 'manual'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None

# Function to preprocess data (used for both manual and file upload)
def preprocess_data(df):
    try:
        df = df.copy()
        
        # Ensure numeric conversion for numerical columns
        for col in numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Normalize categorical columns before mapping
        for col in ['Sleep_Duration', 'Dietary_Habits', 'Have_you_ever_had_suicidal_thoughts_', 'Family_History_of_Mental_Illness']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.capitalize()

        # Handle categorical mappings
        if 'Sleep_Duration' in df.columns:
            df['Sleep_Duration'] = df['Sleep_Duration'].map(sleep_duration_mapping)
        if 'Dietary_Habits' in df.columns:
            df['Dietary_Habits'] = df['Dietary_Habits'].map(dietary_habits_mapping)
        if 'Have_you_ever_had_suicidal_thoughts_' in df.columns:
            df['Have_you_ever_had_suicidal_thoughts_'] = df['Have_you_ever_had_suicidal_thoughts_'].map(binary_mapping)
        if 'Family_History_of_Mental_Illness' in df.columns:
            df['Family_History_of_Mental_Illness'] = df['Family_History_of_Mental_Illness'].map(binary_mapping)

        # Replace -1 with a default valid value for prediction
        for col in ['Sleep_Duration', 'Dietary_Habits']:
            if col in df.columns:
                if (df[col] == -1).any():
                    # Use a sensible default if mode can't be computed (e.g., all -1 or empty)
                    valid_values = df[col][df[col] != -1]
                    default_value = valid_values.mode()[0] if not valid_values.empty else (2 if col == 'Sleep_Duration' else 1)
                    df[col] = df[col].replace(-1, default_value)

        # Check for unmapped values (NaN) and show specific invalid values
        for col in ['Sleep_Duration', 'Dietary_Habits', 'Have_you_ever_had_suicidal_thoughts_', 'Family_History_of_Mental_Illness']:
            if col in df.columns and df[col].isnull().any():
                invalid_values = df[df[col].isnull()][col].unique()
                st.error(f"Invalid values in column '{col}': {invalid_values}. Expected values: {list(eval(f'{col.lower()}_mapping').keys())}")
                return None

        # Fill missing values
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        for col in ordinal_columns + binary_columns:
            if col in df.columns:
                # Use mode, but fallback to 0 if all values are NaN
                mode_val = df[col].mode()[0] if not df[col].isna().all() else 0
                df[col] = df[col].fillna(mode_val)

        # Ensure correct types
        for col in ordinal_columns + binary_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Align with feature_names
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        return df
    except Exception as e:
        # Provide detailed error message
        import traceback
        st.error(f"Error in data preprocessing: {str(e)}. Details: {traceback.format_exc()}")
        return None

# Function to display the input form for manual input
def show_manual_input_form():
    st.title("Manual Input Mode")

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            work_study_hours = col1.slider("Work/Study Hours per Week", min_value=0, max_value=80, value=40)
            academic_pressure = col1.slider("Academic Pressure (0 = None, 5 = Very High)", min_value=0, max_value=5, value=0)
            study_satisfaction = col1.slider("Study Satisfaction (0 = None, 5 = Immense)", min_value=0, max_value=5, value=0)
            financial_stress = col1.slider("Financial Stress (0 = None, 5 = Very High)", min_value=0, max_value=5, value=0)
        
        with col2:
            sleep_duration = col2.selectbox("Sleep Duration", options=['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours', 'Others'])
            dietary_habits = col2.selectbox("Dietary Habits", options=['Healthy', 'Moderate', 'Unhealthy', 'Others'])
            suicidal_thoughts = col2.radio("Have you ever had suicidal thoughts?", options=['No', 'Yes'])
            family_history = col2.radio("Family History of Mental Illness?", options=['No', 'Yes'])
        
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        user_input = {
            'Work_Study_Hours': work_study_hours,
            'Academic_Pressure': academic_pressure,
            'Study_Satisfaction': study_satisfaction,
            'Financial_Stress': financial_stress,
            'Sleep_Duration': sleep_duration,
            'Dietary_Habits': dietary_habits,
            'Have_you_ever_had_suicidal_thoughts_': suicidal_thoughts,
            'Family_History_of_Mental_Illness': family_history
        }

        user_df = pd.DataFrame([user_input])
        process_and_predict(user_df)

# Function to display the file upload form
def show_file_upload_form():
    st.title("File Upload Mode")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.uploaded_data = pd.read_csv(uploaded_file)
            else:
                st.session_state.uploaded_data = pd.read_excel(uploaded_file)
            st.write("File uploaded successfully!")
            if st.button("Predict from Uploaded Data"):
                predict_from_uploaded_data()
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

# Function to process and predict from uploaded data
def predict_from_uploaded_data():
    uploaded_df = st.session_state.uploaded_data.copy()
    
    # Check for required columns
    missing_cols = [col for col in relevant_columns if col not in uploaded_df.columns]
    if missing_cols:
        st.error(f"Missing required columns in uploaded file: {missing_cols}")
        return
    
    # Check for extra columns (excluding 'Depression')
    extra_cols = [col for col in uploaded_df.columns if col not in relevant_columns and col != 'Depression']
    if extra_cols:
        # st.warning(f"Extra columns found and will be ignored: {extra_cols}")
        uploaded_df = uploaded_df[relevant_columns + (['Depression'] if 'Depression' in uploaded_df.columns else [])]
    
    # Preprocess the data
    processed_df = preprocess_data(uploaded_df)
    if processed_df is None:
        return
    
    # Check for Depression column for accuracy calculation
    actual_labels = uploaded_df['Depression'] if 'Depression' in uploaded_df.columns else None
    
    # Make predictions
    threshold = 0.5
    prediction_proba = model.predict_proba(processed_df)
    predictions = (prediction_proba[:, 1] >= threshold).astype(int)
    prediction_labels = ["Yes" if p == 1 else "No" for p in predictions]

    # Display results
    result_df = pd.DataFrame({'Prediction': prediction_labels})
    
    if actual_labels is not None:
        # Check for invalid values in 'Depression' column
        invalid_values = actual_labels[~actual_labels.isin([0, 1])].unique()
        if len(invalid_values) > 0:
            st.warning(f"Invalid values in 'Depression' column: {invalid_values}. Expected 0 or 1. These rows will be ignored for accuracy calculation.")
            valid_indices = actual_labels.isin([0, 1])
            actual_binary = actual_labels[valid_indices]
            predictions_valid = np.array(predictions)[valid_indices]
            prediction_labels = np.array(prediction_labels)[valid_indices]
        else:
            actual_binary = actual_labels
            predictions_valid = predictions
            
        if len(actual_binary) > 0:
            st.session_state.accuracy = accuracy_score(actual_binary, predictions_valid)
            st.write(f"**Accuracy: {st.session_state.accuracy:.2%}**")
        else:
            st.warning("No valid values in 'Depression' column for accuracy calculation.")
            
        result_df['Actual'] = actual_labels.values
    
    # Display the predictions table with custom styling
    st.write("**Predictions:**")
    st.markdown(
        """
        <style>
        .dataframe {
            font-size: 18px !important;
            width: 100% !important;
        }
        .dataframe th, .dataframe td {
            padding: 15px !important;
            text-align: center !important;
        }
        .dataframe th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        .dataframe tr {
            height: 50px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(result_df, use_container_width=True)

# Function to process and predict (for manual input)
def process_and_predict(df, return_prediction=False):
    try:
        # Preprocess the data
        processed_df = preprocess_data(df)
        if processed_df is None:
            return None

        # Prediction
        threshold = 0.5
        prediction_proba = model.predict_proba(processed_df)
        prediction = (prediction_proba[:, 1] >= threshold).astype(int)
        
        st.session_state.prediction = "Yes" if prediction[0] == 1 else "No"
        
        if return_prediction:
            return st.session_state.prediction
        
        st.session_state.view = 'warning' if st.session_state.prediction == "Yes" else 'safe'
        st.rerun()
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

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
        .spacer {
            margin-top: 40px;  /* Adds space before the button */
        }
        </style>
        <div class="warning-box">
            <h2 style="color: red;">⚠️ Warning: Depression Tendency Detected</h2>
            <p class="warning-text">The model predicts you might be experiencing depression. Please consider consulting a mental health professional.</p>
        </div>
        <div class="spacer"></div>  <!-- Spacer added before button -->
        """, 
        unsafe_allow_html=True
    )
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
        .spacer {
            margin-top: 40px;  /* Adds space before the button */
        }
        </style>
        <div class="safe-box">
            <h2 style="color: green;">✅ Safe: No Depression Tendency</h2>
            <p class="safe-text">The model predicts you are not experiencing depression. Stay well!</p>
        </div>
        <div class="spacer"></div>  <!-- Spacer added before button -->
        """, 
        unsafe_allow_html=True
    )
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
            © 2025 Wenbo (Daniel) Zhu. All rights reserved.
        </div>
        """, 
        unsafe_allow_html=True
    )

# Main app layout with elegant mode selection
def main():
    # Custom CSS for better layout and styling
    st.markdown("""
        <style>
        /* Ensure the mode container and content do not stretch too wide */
        .mode-container, .main-content {
            max-width: 800px;  /* Adjust width as needed */
            margin: auto;       /* Center content */
        }
        
        /* Remove unnecessary padding/margin */
        .mode-button {
            margin: 5px 0 !important;
        }

        /* Fix Streamlit's default full-width behavior */
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)


    # Title
    st.title("Depression Prediction Tool")

    # Mode selection with full-width buttons
    with st.container():
        st.markdown('<div class="mode-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Manual Input Mode", 
                        key="manual_mode", 
                        help="Enter data manually using sliders and selections",
                        use_container_width=True):
                st.session_state.mode = 'manual'
                st.session_state.view = 'input'
                st.session_state.prediction = None
                st.session_state.uploaded_data = None
                st.rerun()

        with col2:
            if st.button("File Upload Mode", 
                        key="file_mode", 
                        help="Upload a CSV or Excel file with data",
                        use_container_width=True):
                st.session_state.mode = 'file'
                st.session_state.view = 'input'
                st.session_state.prediction = None
                st.session_state.uploaded_data = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        
        if st.session_state.mode == 'manual' and st.session_state.view == 'input':
            show_manual_input_form()
        elif st.session_state.mode == 'file' and st.session_state.view == 'input':
            show_file_upload_form()
        elif st.session_state.view == 'warning':
            show_warning()
        elif st.session_state.view == 'safe':
            show_safe()
            
        st.markdown('</div>', unsafe_allow_html=True)

    show_copyright()

if __name__ == "__main__":
    main()