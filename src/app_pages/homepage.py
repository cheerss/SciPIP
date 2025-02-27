
import streamlit as st
from .locale import _
from .sidebar_components import get_sidebar_header, get_sidebar_supported_fields, get_help_us_improve, get_language_select

def generate_sidebar():
    get_language_select()
    get_sidebar_header()
    st.sidebar.markdown(_("Make AI research easy"))
    get_sidebar_supported_fields()
    get_help_us_improve()


def generate_mainpage():
    if st.session_state.get("language", "en") == "en":
        st.title("🏠️ 💡SciPIP: An LLM-based Scientific Paper Idea Proposer")
        _, logo_col, _ = st.columns(3)
        logo_col.image("./assets/pic/logo.svg", width=None)

        st.header("Introduction", divider="blue")
        st.markdown("SciPIP is a scientific paper idea generation tool powered by a large language model (LLM) designed to **assist researchers in quickly generating novel research ideas**. Based on the background information provided by the user, SciPIP first conducts a literature review to identify relevant research, then generates fresh ideas for potential studies.")

        st.header("Pipeline", divider="blue")
        _, idea_proposal_col, _ = st.columns([1, 5, 1])
        idea_proposal_col.image("./assets/pic/figure_idea_proposal.svg", width=None)
        st.markdown("""This demo uses SciPIP-C, as described in the [paper](https://arxiv.org/abs/2410.23166), as the default idea generation method. The generation process is mainly divided into six steps:

    1. **Input Background**: The user inputs the background of the research.
    2. **Brainstorming**: The large model, without retrieving any literature, generates solutions to the problems in the user-inputted background based solely on its own knowledge.
    3. **Extracting Entities**: Extract keywords from the user’s input background and the content generated during brainstorming.
    4. **Retrieving Related Works**: Search for relevant literature in the database based on the extracted keywords and the user’s input background.
    5. **Generating Initial Ideas**: Draw inspiration from the retrieved literature and, combined with the brainstorming content, propose initial ideas.
    6. **Generating Final Ideas**: Filter, refine, and process the initial ideas to produce the final ideas.
    """)

        st.header("One-click Generation vs. Step-by-step Generation", divider="blue")
        # st.markdown("一键生成与逐步生成均使用相同的算法（SciPIP-C），对于一键生成而言，用户无需关心所有的中间输出，可以直接得到最终的Ideas。而逐步生成会按照Pipeline的步骤逐步生成，每步生成结束后，用户都可以修订此步骤生成的内容，从而影响后续生成结果。")
        st.markdown("Both one-click generation and step-by-step generation use the same algorithm (SciPIP-C). For one-click generation, the user does not need to concern themselves with the intermediate outputs and can directly obtain the final ideas. In contrast, step-by-step generation follows the pipeline process, where the content is generated step by step. After each step, the user can revise the content generated in that step, which will influence the results of subsequent steps.")

        st.header("Resources")
        st.markdown("Our paper: [https://arxiv.org/abs/2410.23166](https://arxiv.org/abs/2410.23166)")
        st.markdown("Our github repository: [https://github.com/cheerss/SciPIP](https://github.com/cheerss/SciPIP)")
        st.markdown("Our Huggingface demo: [https://huggingface.co/spaces/lihuigu/SciPIP](https://huggingface.co/spaces/lihuigu/SciPIP)")
        # st.page_link("https://arxiv.org/abs/2410.23166", label="Our paper: https://arxiv.org/abs/2410.23166", icon=None)
        # st.page_link("https://github.com/cheerss/SciPIP", label="Our github repository: https://github.com/cheerss/SciPIP", icon=None)

    else:
        st.title("🏠️ 💡SciPIP: 基于大语言模型的科学论文创意生成器")
        _, logo_col, _ = st.columns(3)
        logo_col.image("./assets/pic/logo.svg", width=None)

        st.header("简介", divider="blue")
        st.markdown("SciPIP 是一个由大语言模型（LLM）驱动的科学论文创意生成工具，旨在**帮助研究人员快速生成新颖的研究思路**。基于用户提供的背景信息，SciPIP首先进行文献回顾以识别相关研究，然后为潜在的研究方向生成新的创意。")

        st.header("流程", divider="blue")
        _, idea_proposal_col, _ = st.columns([1, 5, 1])
        idea_proposal_col.image("./assets/pic/figure_idea_proposal.svg", width=None)
        st.markdown("""本演示采用论文中所述的SciPIP-C作为默认的创意生成方法，生成流程主要分为六个步骤：

1. **输入背景**：用户输入研究的背景信息。
2. **头脑风暴**：大模型在不检索任何文献的情况下，仅凭自身知识为用户输入的背景中的问题生成解决方案。
3. **提取实体**：从用户输入的背景和头脑风暴生成的内容中提取关键词。
4. **检索相关文献**：根据提取的关键词和用户输入的背景信息，在数据库中检索相关文献。
5. **生成初始创意**：从检索到的文献中汲取灵感，并结合头脑风暴的内容提出初步创意。
6. **生成最终创意**：对初始创意进行筛选、精炼和加工，最终生成创意。
    """)

        st.header("一键生成 与 逐步生成", divider="blue")
        st.markdown("一键生成与逐步生成均使用相同的算法（SciPIP-C），对于一键生成而言，用户无需关心所有的中间输出，可以直接得到最终的Ideas。而逐步生成会按照Pipeline的步骤逐步生成，每步生成结束后，用户都可以修订此步骤生成的内容，从而影响后续生成结果。")

        st.header("相关资源")
        st.markdown("论文: [https://arxiv.org/abs/2410.23166](https://arxiv.org/abs/2410.23166)")
        st.markdown("Github仓库: [https://github.com/cheerss/SciPIP](https://github.com/cheerss/SciPIP)")
        st.markdown("Huggingface演示: [https://huggingface.co/spaces/lihuigu/SciPIP](https://huggingface.co/spaces/lihuigu/SciPIP)")
        # st.page_link("https://arxiv.org/abs/2410.23166", label="Our paper: https://arxiv.org/abs/2410.23166", icon=None)
        # st.page_link("https://github.com/cheerss/SciPIP", label="Our github repository: https://github.com/cheerss/SciPIP", icon=None)

def home_page():
    generate_sidebar()
    generate_mainpage()