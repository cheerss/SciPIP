import streamlit as st
from loguru import logger
from .locale import _
from .sidebar_components import get_sidebar_header, get_sidebar_supported_fields, get_help_us_improve, get_language_select

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
    st.title(_("One-click Generation"))

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
    with st.spinner(text=("Extracting entities from the user's input...")):
        entities_bg = backend.background2entities_callback(background)

    with st.spinner(text=("Understanding the user's input...")):
        expanded_background = backend.background2expandedbackground_callback(background, entities_bg)
        st.session_state["intermediate_output"]["expanded_background"] = {"role": "assistant", "content": expanded_background}

    with st.spinner(text=("Brainstorming...")):
        brainstorms = backend.background2brainstorm_callback(expanded_background)
        st.session_state["intermediate_output"]["brainstorms"] = {"role": "assistant", "content": brainstorms}
        st.chat_message("assistant").write("I have the following thoughts, but I'll search the literature to further consolidate and improve the ideas.")
        st.chat_message("assistant").write(brainstorms)

    with st.spinner(text=("Extracting entities for literature retrieval...")):
        entities_all = backend.brainstorm2entities_callback(brainstorms, entities_bg)
        st.session_state["intermediate_output"]["entities"] = {"role": "assistant", "content": entities_all}
        # st.chat_message("assistant").write(entities)

    with st.spinner(text=("Retrieving related works...")):
        msg = "The retrieved works include:"
        related_works, related_works_intact = backend.entities2literature_callback(expanded_background, entities_all)
        st.session_state["intermediate_output"]["related_works"] = {"role": "assistant", "content": related_works}
        # st.chat_message("assistant").write(related_works)

    with st.spinner(text="Generating ideas... (This may take up to 5 minutes)"):
        initial_ideas, final_ideas = backend.literature2initial_ideas_callback(background, brainstorms, related_works_intact)
        logger.info(f"Num of initial ideas: {len(initial_ideas)}, num of final ideas: {len(final_ideas)}")
        # assert len(initial_ideas) == len(final_ideas)
        msg = f"I have {len(initial_ideas)} ideas:"
        st.chat_message("assistant").write(msg)
        for i in range(len(initial_ideas)):
            output = f"""### Concise Idea
{initial_ideas[i]}

### Idea in Detail:

{final_ideas[i]}

"""
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.chat_message("assistant").write(output)

def one_click_generation(backend):
    generate_sidebar()
    generate_mainpage(backend)