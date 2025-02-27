import ast
import streamlit as st
from .locale import _
from .sidebar_components import get_sidebar_header, get_sidebar_supported_fields, get_help_us_improve, get_language_select

def generate_sidebar():
    get_language_select()
    get_sidebar_header()
    st.sidebar.markdown(
        _("SciPIP will generate ideas step by step. The generation pipeline is the same as "
        "one-click generation, while you can improve each part manually after SciPIP providing the manuscript.")
    )

    DONE_COLOR = "black"
    UNDONE_COLOR = "gray"
    # INPROGRESS_COLOR = "#4d9ee6"
    INPROGRESS_COLOR = "black"
    color_list = []
    pipeline_list = [_("1. Input Background"), _("2. Brainstorming"), _("3. Extracting Entities"), _("4. Retrieving Related Works"), 
                     _("5. Generating Initial Ideas"), _("6. Generating Final Ideas")]
    for i in range(1, 8):
        if st.session_state["global_state_step"] < i:
            color_list.append(UNDONE_COLOR)
        elif st.session_state["global_state_step"] == i:
            color_list.append(INPROGRESS_COLOR)
        elif st.session_state["global_state_step"] > i:
            color_list.append(DONE_COLOR)
    st.sidebar.header(_("Pipeline"), divider="red")
    for i in range(6):
        st.sidebar.markdown(f"<font color='{color_list[i]}'>{pipeline_list[i]}</font>", unsafe_allow_html=True)
        # if st.session_state["global_state_step"] == i + 1:
        #     st.sidebar.progress(50, text=None)

    get_sidebar_supported_fields()
    get_help_us_improve()

def get_textarea_height(text_content):
    if text_content is None:
        return 100
    lines = text_content.split("\n")
    count = len(lines)
    for line in lines:
        count += len(line) // 96
    return max(count * 23 + 20, 100) # 23 is a magic number

def generate_mainpage(backend):
    st.title(_("Step-by-step Generation"))
    st.header(_("Background"))
    with st.form('background_form') as bg_form:
        background = st.session_state.get("background", "")
        background = st.text_area("Input your field background", background, placeholder="Input your field background", height=200, label_visibility="collapsed")

        cols = st.columns(4)
        def click_demo_i(i):
            st.session_state["background"] = backend.get_demo_i(i)
        for i, col in enumerate(cols):
            col.form_submit_button(_("Example") + f" {i+1}", use_container_width=True, on_click=click_demo_i, args=(i,))

        col1, col2 = st.columns([2, 20])
        submitted = col1.form_submit_button(_("Submit"), type="primary")
        if submitted:
            st.session_state["global_state_step"] = 2.0
            with st.spinner(text="Let me first brainstorm some ideas..."):
                st.session_state["entities_bg"] = backend.background2entities_callback(background)
                st.session_state["expanded_background"] = backend.background2expandedbackground_callback(
                    background, st.session_state["entities_bg"]
                )
                st.session_state["brainstorms"] = backend.background2brainstorm_callback(
                    st.session_state["expanded_background"]
                )
            # st.session_state["brainstorms"] = "Test text"
            st.session_state["brainstorms_expand"] = True
            st.session_state["global_state_step"] = 2.5

    ## Brainstorms
    if st.session_state["global_state_step"] >= 2.5:
        st.header(_("Brainstorms"))
        with st.expander("", expanded=st.session_state.get("brainstorms_expand", False)):
            # st.write("<div class='myclass'>")
            col1, col2 = st.columns(2)
            widget_height = get_textarea_height(st.session_state.get("brainstorms", ""))
            brainstorms = col1.text_area(label="brainstorms", value=st.session_state.get("brainstorms", ""), 
                                        label_visibility="collapsed", height=widget_height)
            st.session_state["brainstorms"] = brainstorms
            if brainstorms:
                col2.markdown(f"{brainstorms}")
            else:
                col2.markdown(_("Please input the brainstorms on the left."))
            # st.write("</div>")
            col1, col2 = st.columns([2, 20])
            submitted = col1.button(_("Submit"), type="primary")
            if submitted:
                st.session_state["global_state_step"] = 3.0
                with st.spinner(text="I'am extracting keywords in the background and brainstorming ideas"):
                    st.session_state["entities"] = backend.brainstorm2entities_callback(brainstorms, st.session_state["entities_bg"])
                # st.session_state["entities"] = "entities"
                st.session_state["global_state_step"] = 3.5
                st.session_state["entities_expand"] = True

    ## Entities
    if st.session_state["global_state_step"] >= 3.5:
        st.header(_("Extracted Entities"))
        with st.expander("", expanded=st.session_state.get("entities_expand", False)):
            ## pills
            def update_entities():
                return
            ori_entities = st.session_state.get("entities", [])
            entities_updated = st.session_state.get("entities_updated", ori_entities)
            entities_updated = st.pills(label="entities", options=ori_entities, selection_mode="multi", 
                                default=ori_entities, label_visibility="collapsed", on_change=update_entities)
            st.session_state["entities_updated"] = entities_updated

            submitted = st.button(_("Submit"), key="entities_button", type="primary")
            if submitted:
                st.session_state["global_state_step"] = 4.0
                with st.spinner(text="I am retrieving related works for more ideas..."):
                    st.session_state["related_works"], st.session_state["related_works_intact"] = \
                        backend.entities2literature_callback(st.session_state["expanded_background"], entities_updated)
                st.session_state["related_works_use_state"] = [True] * len(st.session_state["related_works"])
                st.session_state["global_state_step"] = 4.5
                st.session_state["related_works_expand"] = True

    ## Retrieved related works
    if st.session_state["global_state_step"] >= 4.5:
        st.header(_("Retrieved Related Works"))
        with st.expander("", expanded=st.session_state.get("related_works_expand", False)):
            related_works = st.session_state.get("related_works", [])
            for i, rw in enumerate(related_works):
                checked = st.checkbox(rw, value=st.session_state.get("related_works_use_state")[i])
                st.session_state.get("related_works_use_state")[i] = checked

            submitted = st.button(_("Submit"), key="related_works_button", type="primary")
            if submitted:
                st.session_state["global_state_step"] = 5.0
                with st.spinner(text="I am generating final ideas..."):
                    st.session_state["selected_related_works_intact"] = []
                    for s, p in zip(st.session_state.get("related_works_use_state"), st.session_state["related_works_intact"]):
                        if s:
                            st.session_state["selected_related_works_intact"].append(p)
                    res = backend.literature2initial_ideas_callback(background, brainstorms, st.session_state["selected_related_works_intact"])
                    st.session_state["initial_ideas"] = res[0]
                    st.session_state["final_ideas"] = res[1]
                # st.session_state["initial_ideas"] = "initial ideas"
                st.session_state["global_state_step"] = 5.5
                st.session_state["initial_ideas_expand"] = True

    ## Initial ideas
    if st.session_state["global_state_step"] >= 5.5:
        st.header(_("Generated Ideas"))
        with st.expander("", expanded=st.session_state.get("initial_ideas_expand", False)):
            for initial_idea, final_idea in zip(st.session_state.get("initial_ideas", ""), st.session_state.get("final_ideas", "")):
                st.divider()
                st.markdown("### Concise Idea")
                st.markdown(initial_idea)
                st.markdown("### Idea in Detail")
                st.markdown(final_idea)
                st.divider()

def step_by_step_generation(backend):
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
    if "global_state_step" not in st.session_state:
        st.session_state["global_state_step"] = 1.0
    # backend = button_interface.Backend()
    generate_mainpage(backend)
    generate_sidebar()
