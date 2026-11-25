
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------
# CUSTOM CSS DESIGN
# ------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
    }

    h1 {
        color: #2563eb;
        text-align: center;
    }

    h2, h3 {
        color: #1e40af;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 18px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #bfdbfe;
    }

    .result-box {
        background-color: #dbeafe;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #93c5fd;
    }

    .result-title {
        color: #1d4ed8;
        font-size: 24px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# LOAD MACHINE LEARNING MODEL
# ------------------------------------------

MODEL_PATH = Path(__file__).parent / "model.pkl"

if not MODEL_PATH.exists():
    st.error("Model file not found!")

    st.warning(
        "Please run train_model.py first "
        "to generate model.pkl."
    )

    st.stop()

model = joblib.load(MODEL_PATH)

# ------------------------------------------
# HEADER
# ------------------------------------------

st.title("🎓 Student Performance Predictor")

st.markdown(
    '<p class="subtitle">Machine Learning Dashboard</p>',
    unsafe_allow_html=True
)

st.write(
    "Predict student performance using "
    "study hours, attendance, and previous scores."
)

st.divider()

# ------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------

st.sidebar.title("📋 Student Information")

student_name = st.sidebar.text_input(
    "Student Name",
    placeholder="Enter student name"
)

study_hours = st.sidebar.slider(
    "Study Hours per Day",
    min_value=0,
    max_value=12,
    value=4
)

attendance = st.sidebar.slider(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=80
)

previous_score = st.sidebar.slider(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=70
)

predict_button = st.sidebar.button(
    "Predict Performance",
    type="primary",
    use_container_width=True
)

# ------------------------------------------
# DASHBOARD METRICS
# ------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Study Hours",
        value=f"{study_hours} hours"
    )

with col2:
    st.metric(
        label="Attendance",
        value=f"{attendance}%"
    )

with col3:
    st.metric(
        label="Previous Score",
        value=f"{previous_score}/100"
    )

st.divider()

# ------------------------------------------
# MACHINE LEARNING PREDICTION
# ------------------------------------------

if predict_button:

    if student_name.strip() == "":
        st.warning("Please enter the student's name.")

    else:

        # Prepare input data
        input_data = pd.DataFrame(
            [[study_hours, attendance, previous_score]],
            columns=[
                "Study Hours",
                "Attendance",
                "Previous Score"
            ]
        )

        # Predict performance
        prediction = model.predict(input_data)[0]

        # Display prediction result
        st.subheader("📊 Prediction Result")

        st.markdown(
            f"""
            <div class="result-box">
                <h2>{student_name}</h2>
                <p class="result-title">
                    Predicted Performance
                </p>
                <h1>{prediction}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # --------------------------------------
        # PREDICTION PROBABILITY
        # --------------------------------------

        probabilities = model.predict_proba(input_data)[0]

        categories = model.classes_

        probability_data = pd.DataFrame({
            "Performance Category": categories,
            "Probability (%)": probabilities * 100
        })

        probability_data = probability_data.sort_values(
            by="Probability (%)",
            ascending=False
        )

        st.subheader("📈 Prediction Confidence")

        st.bar_chart(
            probability_data.set_index(
                "Performance Category"
            )
        )

        # --------------------------------------
        # RESULT TABLE
        # --------------------------------------

        st.subheader("📄 Student Result Table")

        result_table = pd.DataFrame({
            "Student Name": [student_name],
            "Study Hours": [study_hours],
            "Attendance (%)": [attendance],
            "Previous Score": [previous_score],
            "Predicted Performance": [prediction]
        })

        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------
        # DOWNLOAD RESULT
        # --------------------------------------

        csv_data = result_table.to_csv(index=False)

        st.download_button(
            label="Download Student Result",
            data=csv_data,
            file_name="student_performance_dataset.csv",
            mime="text/csv"
        )

else:

    st.info(
        "Enter the student's information in the sidebar "
        "and click Predict Performance."
    )

# ------------------------------------------
# FOOTER
# ------------------------------------------

st.divider()

st.caption(
    "Student Performance Predictor | "
    "Python | Scikit-learn | Streamlit"
)
