import json
import streamlit as st

json_contents = json.loads(open("./src/app_pages/locale.json", "r").read())

def _(content: str):
    if st.session_state.get("language", "en") == "en":
        return content
    a = json_contents.get(content, content)
    if isinstance(a, dict):
        return a.get(st.session_state["language"], content)
    else:
        return content

