
# Core Pkgs
import streamlit as st
import altair as alt

# EDA Pkgs
import pandas as pd 
import numpy as np 
#Utils
import joblib
from PIL import Image
image = Image.open('logo.png')
cola, colb, colc = st.columns([3,6,1])
with cola:
    st.write("")

with colb:
    st.image(image, width = 300)

with colc:
    st.write("")



pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))

#emotions emoji dictionery
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Fxn

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results
def main():
    st.title("Emotion Classifier Interface")
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
        if submit_text:
            col1,col2  = st.columns(2)
            # Apply Fxn Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)


    elif choice == "Monitor":
        st.subheader("Monitor App")
        st.write("Under Construction")
    else:
        st.subheader("About")
        
        st.write("Under Construction")


if __name__ == '__main__':
    main()