import streamlit as st
from .locale import _

# def generate_sidebar():

def get_sidebar_header():
    st.sidebar.header("SciPIP", divider="rainbow")

def get_sidebar_supported_fields():
    st.sidebar.header(_("Supported Fields"), divider="orange")
    st.sidebar.caption(_("The supported fields are temporarily limited because we only collect literature "
               "from ICML, ICLR, NeurIPS, ACL, and EMNLP. Support for other fields are in progress."))
    st.sidebar.checkbox(_("Natural Language Processing (NLP)"), value=True, disabled=True)
    st.sidebar.checkbox(_("Computer Vision (CV)"), value=True, disabled=True)

    st.sidebar.checkbox(_("Multimodal"), value=True, disabled=True)
    st.sidebar.checkbox(_("Incoming Other Fields"), value=False, disabled=True)

def get_help_us_improve():
    st.sidebar.header(_("Help Us To Improve"), divider="green")
    st.sidebar.markdown("https://forms.gle/YpLUrhqs1ahyCAe99", unsafe_allow_html=True)

def get_language_select():
    language = st.session_state.get("language", "en")
    language_option = st.sidebar.segmented_control(
        "选择语言 / Select Language",
        options=["中文", "English"],
        selection_mode="single",
        default=("中文" if language == "zh" else "English")
    )
    if language_option == "中文":
        language = "zh"
    elif language_option == "English":
        language = "en"
    if language != st.session_state.get("language", "en"):
        st.session_state["language"] = language
        st.rerun()