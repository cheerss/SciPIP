import streamlit as st
from .locale import _
from .sidebar_components import get_sidebar_header, get_sidebar_supported_fields, get_help_us_improve, get_language_select

# st.set_page_config(layout="wide", page_title="🦜🔗 Generate Idea Step-by-step")

## Pipeline global state
# 1.0: Input background is in progress
# 2.0: Brainstorming is in progress
#  2.5 Brainstorming is finished
# 3.0: Extracting entities is in progress
#  3.5 Extracting entities is finished
# 4.0: Retrieving literature is in progress
#  4.5 Retrieving ideas is finished
# 5.0: Generating initial ideas is in progress
#  5.5 Generating initial ideas is finished
# 6.0: Generating final ideas is in progress
#  6.5 Generating final ideas is finished
if "global_state_one_click" not in st.session_state:
    st.session_state["global_state_one_click"] = 1.0

def generate_sidebar():
    get_language_select()
    get_sidebar_header()
    st.sidebar.markdown(
        _("SciPIP will generate ideas in one click. The generation pipeline is the same as "
        "step-by-step generation, but you are free from caring about intermediate outputs.")
    )

    pipeline_list = [_("1. Input Background"), _("2. Brainstorming"), _("3. Extracting Entities"), _("4. Retrieving Related Works"), 
                     _("5. Generating Initial Ideas"), _("6. Generating Final Ideas")]
    st.sidebar.header(_("Pipeline"), divider="red")
    for i in range(6):
        st.sidebar.markdown(f"<font color='black'>{pipeline_list[i]}</font>", unsafe_allow_html=True)

    get_sidebar_supported_fields()
    get_help_us_improve()

def generate_mainpage(backend):
    st.title(_("💧 One-click Generation"))

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Please give me some key words or a background"}]
    if "intermediate_output" not in st.session_state:
        st.session_state["intermediate_output"] = {}

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    def disable_submit():
        st.session_state["enable_submmit"] = False

    if prompt := st.chat_input(disabled=not st.session_state.get("enable_submmit", True), on_submit=disable_submit):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        generate_ideas(backend, prompt)
    elif st.session_state.get("use_demo_input", False):
        generate_ideas(backend, st.session_state.get("demo_input"))
        st.session_state["use_demo_input"] = False
        del(st.session_state["demo_input"])

    def get_demo_n(i):
        demo_input = backend.get_demo_i(i)
        st.session_state["enable_submmit"] = False
        st.session_state.messages.append({"role": "user", "content": demo_input})
        st.session_state["use_demo_input"] = True
        st.session_state["demo_input"] = demo_input

    cols = st.columns([1, 1, 1, 1])
    for i in range(4):
        cols[i].button(_("Example") + f" {i+1}", on_click=get_demo_n, args=(i,), use_container_width=True, disabled=not st.session_state.get("enable_submmit", True))
    # cols[0].button(_("Example 1"), on_click=get_demo_n, args=(0,), use_container_width=True, disabled=not st.session_state.get("enable_submmit", True))
    # cols[1].button(_("Example 2"), on_click=get_demo_n, args=(1,), use_container_width=True, disabled=not st.session_state.get("enable_submmit", True))
    # cols[2].button(_("Example 3"), on_click=get_demo_n, args=(2,), use_container_width=True, disabled=not st.session_state.get("enable_submmit", True))
    # cols[3].button(_("Example 4"), on_click=get_demo_n, args=(3,), use_container_width=True, disabled=not st.session_state.get("enable_submmit", True))

    def check_intermediate_outputs(id="brainstorms"):
        msg = st.session_state["intermediate_output"].get(id, None)
        if msg is not None:
            st.session_state.messages.append(msg)
        else:
            st.toast(f"No {id} now!")

    def reset():
        del(st.session_state["messages"])
        del(st.session_state["intermediate_output"])
        st.session_state["enable_submmit"] = True
        st.session_state["global_state_one_click"] = 1.0
        st.toast(f"The chat has been reset!")

    cols = st.columns([1, 1, 1, 1])
    cols[0].button(_("Check Brainstorms"), on_click=check_intermediate_outputs, args=("brainstorms",), use_container_width=True)
    cols[1].button(_("Check Entities"), on_click=check_intermediate_outputs, args=("entities",), use_container_width=True)
    cols[2].button(_("Check Retrieved Papers"), on_click=check_intermediate_outputs, args=("related_works",), use_container_width=True)
    cols[3].button(_("Reset Chat"), on_click=reset, use_container_width=True, type="primary")

def generate_ideas(backend, background):
    with st.spinner(text=("Brainstorming...")):
        brainstorms = backend.background2brainstorm_callback(background)
        st.session_state["intermediate_output"]["brainstorms"] = {"role": "assistant", "content": brainstorms}
        # st.chat_message("assistant").write(brainstorms)
        st.session_state["global_state_one_click"] = 2.5

    with st.spinner(text=("Extracting entities...")):
        entities = backend.brainstorm2entities_callback(background, brainstorms)
        st.session_state["intermediate_output"]["entities"] = {"role": "assistant", "content": entities}
        # st.chat_message("assistant").write(entities)
        st.session_state["global_state_one_click"] = 3.5

    with st.spinner(text=("Retrieving related works...")):
        msg = "My initial ideas are:"
        related_works, related_works_intact = backend.entities2literature_callback(background, entities)
        st.session_state["intermediate_output"]["related_works"] = {"role": "assistant", "content": related_works}
        # st.chat_message("assistant").write(related_works)
        st.session_state["global_state_one_click"] = 4.5

    with st.spinner(text="Generating initial ideas..."):
        msg = "My initial ideas are:"
        initial_ideas, final_ideas = backend.literature2initial_ideas_callback(background, brainstorms, related_works_intact)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        st.session_state.messages.append({"role": "assistant", "content": initial_ideas})
        st.chat_message("assistant").write(initial_ideas)
        st.session_state["global_state_one_click"] = 5.5

    with st.spinner(text=("Generating final ideas...")):
        msg = "My final ideas after refinement are:"
        final_ideas = backend.initial2final_callback(initial_ideas, final_ideas)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        st.session_state.messages.append({"role": "assistant", "content": final_ideas})
        st.chat_message("assistant").write(final_ideas)
        st.session_state["global_state_one_click"] = 6.5

def one_click_generation(backend):
    generate_sidebar()
    generate_mainpage(backend)