import sys

sys.path.append("./src")
import streamlit as st
from app_pages import (
    button_interface,
    step_by_step_generation,
    one_click_generation,
    homepage,
)
from app_pages.locale import _

if __name__ == "__main__":
    backend = button_interface.Backend()
    # backend = None
    st.set_page_config(layout="wide")

    # st.logo("./assets/pic/logo.jpg", size="large")
    def fn1():
        one_click_generation.one_click_generation(backend)

    def fn2():
        step_by_step_generation.step_by_step_generation(backend)

    pg = st.navigation([
        st.Page(homepage.home_page, title=_("🏠️ Homepage")),
        st.Page(fn1, title=_("💧 One-click Generation")),
        st.Page(fn2, title=_("💦 Step-by-step Generation")),
    ])
    pg.run()