import pickle
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
import string
from textblob import Word
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from textblob import TextBlob

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")


def transform_text(txt_input):
    txt_input = txt_input.lower()
    txt_input = ''.join((x for x in txt_input if not x.isdigit()))
    txt_input = txt_input.translate(str.maketrans('', '', string.punctuation))
    txt_input = word_tokenize(txt_input)
    txt_input = [word for word in txt_input if not word in stopwords.words()]
    txt_input = (" ").join(txt_input)
    txt_input=txt_input.split(" ")
    txt_input=[Word(word).lemmatize("v") for word in txt_input]
    txt_input = (" ").join(txt_input)
    
    # blob_object = TextBlob(txt_input)
    # txt_input=blob_object.words
    blob_emptyline2 = []
    # for i in text:
    blob = TextBlob(txt_input).sentiment
    # st.write(blob)
    x=blob.polarity
    if x>0:
        st.header("Not depressed")
    if x<0:
        st.header("Depressed")
    else:
        st.header("")
    # blob_emptyline2.append(blob)
    # st.write(blob_emptyline2)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
img_contact_form = Image.open("images/yt_contact_form.png")
img_lottie_animation = Image.open("images/yt_lottie_animation.png")

# ---- HEADER SECTION ----
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:

        st.subheader("Hi, we are DATA GEEKS :wave:")
        st.title("Pre final year students of RGIT")
        st.write(
        "We are trying to link mental health and social media platforms using Python and NLP for analysing mental stability."
    )
        st.write("[Learn More >](https://pythonandvba.com)")
    with right_column:
        st.image("images/tb.png",width=450)
# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("About our project")
        st.write("##")
        st.write(
            """
            Depression Detection:
            - Depression is a serious mental health disorder that affects millions of people worldwide, and early detection is critical to successful treatment.
            - The use of machine learning classifiers in depression detection has become increasingly popular in recent years.
            - Machine learning techniques have shown great promise in the detection and diagnosis of mental health disorders, including depression, by analyzing large datasets of clinical and self-reported measures of symptoms.
            - Our study explores the potential of machine learning classifiers in improving depression detection.
            
            If this sounds interesting to you, you can refer the code given below.
            """
        )
        st.write("[Code link >](https://colab.research.google.com/drive/1XT1BQ9IgxIlvN4rYOaDvKVR8e8e8WLCK#scrollTo=Ky7TQfatzHRw)")
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

# ---- PROJECTS ----
with st.container():
    st.header("Let's Analyse your Mental State")
    txt_input = st.text_input('Enter Your Tweet:', '')
    
    if st.button("Predict"):
        transform_text(txt_input)
        tfidf=pickle.load(open("vectorizer_dd.pkl","rb"))
        model=pickle.load(open("model_dd.pkl","rb"))
        # st.write(txt_input)
        vec_input=tfidf.transform([txt_input])
        result=model.predict(vec_input)[0]
        








# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

