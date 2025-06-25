import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the model
pipe_lr = joblib.load(open("/Users/ernazerkinbekov/Desktop/course_project_ML/Text Emotion Detection/data/text_emotion.pkl", "rb"))
emotions_emoji_dict = {"anger": "1", "disgust": "2", "fear": "3", "happy": "4", "joy": "5", "neutral": "6", "sad": "7",
                       "sadness": "8", "shame": "9", "surprise": "10"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():

    col1, col2= st.columns(2)
    with col1:
        st.title("Tweet emotions detection")


    with col2:
        st.image("/Users/ernazerkinbekov/Desktop/course_project_ML/images/img.png", width=200)

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)

            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)






if __name__ == '__main__':
    main()