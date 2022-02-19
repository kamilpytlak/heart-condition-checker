import streamlit as st
import pandas as pd
import pickle

DATASET_PATH = "data/heart_2020.csv"
LOG_MODEL_PATH = "models/logistic_regression.pkl"

def main():
    @st.cache(persist=True)
    def load_dataset():
        heart = pd.read_csv(DATASET_PATH, encoding="UTF-8")
        return heart

    def user_input_features():
        race = st.sidebar.selectbox("Race", options=(race for race in heart.Race.unique()))
        sex = st.sidebar.selectbox("Sex", options=(sex for sex in heart.Sex.unique()))
        age_cat = st.sidebar.selectbox("Age Category", options=(age_cat for age_cat in heart.AgeCategory.unique()))
        bmi = st.sidebar.number_input("BMI", 0.0, 120.0, 20.0)
        sleep_time = st.sidebar.number_input("Sleep time (in hours)", 0, 24, 8)
        gen_health = st.sidebar.selectbox("General health",
                                          options=(gen_health for gen_health in heart.GenHealth.unique()))
        phys_health = st.sidebar.number_input("Good physical health (in days)", 0, 30, 0)
        ment_health = st.sidebar.number_input("Good mental health (in days)", 0, 30, 0)
        phys_act = st.sidebar.selectbox("Physical activity in the past month", options=("No", "Yes"))
        smoking = st.sidebar.selectbox("Smoking (more than 100 cigarettes in a lifetime)",
                                       options=("No", "Yes"))
        alcohol_drink = st.sidebar.selectbox("Alcohol drinking", options=("No", "Yes"))
        stroke = st.sidebar.selectbox("Stroke", options=("No", "Yes"))
        diff_walk = st.sidebar.selectbox("Difficulty in walking", options=("No", "Yes"))
        diabetic = st.sidebar.selectbox("Diabetic", options=(diabetic for diabetic in heart.Diabetic.unique()))
        asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
        kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
        skin_canc = kid_dis = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))

        features = pd.DataFrame({
            "BMI": [bmi],
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "SleepTime": [sleep_time],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [phys_act],
            "GenHealth": [gen_health],
            "Asthma": [asthma],
            "KidneyDisease": [kid_dis],
            "SkinCancer": [skin_canc]
        })

        return features

    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="images/heart.png"
    )

    st.title("Heart Disease Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="I'll help you diagnose your heart health! - Dr. Logistic Regression",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a logistic regression model using an undersampling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 80%, which is quite good).
        
        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best descibe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.**
        
        You can see the steps of building the model, evaluating it, and cleaning the data
        itself on my GitHub repo: 
        """)

    heart = load_dataset()

    st.sidebar.title("Feature Selection")

    input_df = user_input_features()
    df = pd.concat([input_df, heart], axis=0)
    df = df.drop(columns=["HeartDisease"])

    cat_cols = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
                "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

    df = df[:1]
    df.fillna(0, inplace=True)

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    if submit:
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have heart disease is {round(prediction_prob[0][1] * 100, 2)}%.\n"
                        f"You are healthy.**")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. Logistic Regression",
                     )
        else:
            st.markdown(f"**The probability that you'll have heart disease is {round(prediction_prob[0][1] * 100, 2)}%.\n"
                        f"You are not healthy.**")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the state of your heart! - Dr. Logistic Regression")


if __name__ == "__main__":
    main()
