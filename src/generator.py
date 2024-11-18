from utils.paper_retriever import RetrieverFactory
from utils.paper_client import PaperClient
from utils.llms_api import APIHelper
from utils.header import ConfigReader
from omegaconf import OmegaConf
import click
import json
from loguru import logger
import warnings
import time
import os
from utils.hash import check_env, check_embedding
warnings.filterwarnings("ignore")


def extract_problem(problem, background):
    start_keyword = "**Research Problem**"
    end_keyword = "**Rationales**"
    start_index = problem.find(start_keyword)
    end_index = problem.find(end_keyword)
    if start_index != -1 and end_index != -1:
        research_problem = problem[start_index:end_index].strip()
    else:
        research_problem = background
    return research_problem

class IdeaGenerator:
    def __init__(
        self, config, paper_list: list[dict] = [], cue_words: list = None, brainstorm: str = None
    ) -> None:
        self.api_helper = APIHelper(config)
        self.paper_list = paper_list
        self.cue_words = cue_words
        self.brainstorm = brainstorm

    def generate_with_cue_words(self, background: str):
        problem, message_input = self.api_helper.generate_problem_with_cue_words(
            background, self.paper_list, self.cue_words
        )
        idea = self.api_helper.generate_idea_with_cue_words(
            problem, self.paper_list, self.cue_words
        )
        idea_filtered = self.api_helper.filter_idea(idea, background)
        return message_input, problem, idea, idea_filtered

    def generate_without_cue_words(self, background: str):
        problem, message_input = self.api_helper.generate_problem(
            background, self.paper_list
        )
        idea = self.api_helper.generate_idea(problem, self.paper_list)
        idea_filtered = self.api_helper.filter_idea(idea, background)
        return message_input, problem, idea, idea_filtered

    def generate_with_cue_words_bs(self, background: str):
        problem, message_input = self.api_helper.generate_problem_with_cue_words(
            background, self.paper_list, self.cue_words
        )
        idea = self.api_helper.generate_idea_with_cue_words(
            problem, self.paper_list, self.cue_words
        )
        idea_filtered = self.api_helper.integrate_idea(background, self.brainstorm, idea)
        return message_input, problem, idea, idea_filtered

    def generate_without_cue_words_bs(self, background: str):
        problem, message_input = self.api_helper.generate_problem(
            background, self.paper_list
        )
        idea = self.api_helper.generate_idea(problem, self.paper_list)
        idea_filtered = self.api_helper.integrate_idea(background, self.brainstorm, idea)
        return message_input, problem, idea, idea_filtered

    def generate_with_cue_words_ins(self, background: str):
        problem, message_input = self.api_helper.generate_problem_with_cue_words(
            background, self.paper_list, self.cue_words
        )
        research_problem = extract_problem(problem, background)
        inspirations = []
        for paper in self.paper_list:
            inspiration = self.api_helper.generate_inspiration_with_cue_words(
                research_problem, paper, self.cue_words
            )
            inspirations.append(inspiration)
        idea = self.api_helper.generate_idea_by_inspiration_with_cue_words(
            problem, inspirations, self.cue_words
        )
        idea_filtered = self.api_helper.filter_idea(idea, background)
        return message_input, problem, inspirations, idea, idea_filtered

    def generate_without_cue_words_ins(self, background: str):
        problem, message_input = self.api_helper.generate_problem(
            background, self.paper_list
        )
        research_problem = extract_problem(problem, background)
        inspirations = []
        for paper in self.paper_list:
            inspiration = self.api_helper.generate_inspiration(
                research_problem, paper
            )
            inspirations.append(inspiration)
        idea = self.api_helper.generate_idea_by_inspiration(
            problem, inspirations
        )
        idea_filtered = self.api_helper.filter_idea(idea, background)
        return message_input, problem, inspirations, idea, idea_filtered
    
    def generate_with_cue_words_ins_bs(self, background: str):
        problem, message_input = self.api_helper.generate_problem_with_cue_words(
            background, self.paper_list, self.cue_words
        )
        research_problem = extract_problem(problem, background)
        inspirations = []
        for paper in self.paper_list:
            inspiration = self.api_helper.generate_inspiration_with_cue_words(
                research_problem, paper, self.cue_words
            )
            inspirations.append(inspiration)
        idea = self.api_helper.generate_idea_by_inspiration_with_cue_words(
            problem, inspirations, self.cue_words
        )
        idea_filtered = self.api_helper.integrate_idea(background, self.brainstorm, idea)
        return message_input, problem, inspirations, idea, idea_filtered

    def generate_without_cue_words_ins_bs(self, background: str):
        problem, message_input = self.api_helper.generate_problem(
            background, self.paper_list
        )
        research_problem = extract_problem(problem, background)
        inspirations = []
        for paper in self.paper_list:
            inspiration = self.api_helper.generate_inspiration(
                research_problem, paper
            )
            inspirations.append(inspiration)
        idea = self.api_helper.generate_idea_by_inspiration(
            problem, inspirations
        )
        idea_filtered = self.api_helper.integrate_idea(background, self.brainstorm, idea)
        return message_input, problem, inspirations, idea, idea_filtered

    def generate(
        self,
        background: str,
        mode: str,
        bs_mode: str = None,
        use_cue_words: bool = False,
    ):
        mode_name = None
        if mode == "backtracking":
            mode_name = "Backtrack"
        elif mode == "new_idea":
            mode_name = "Generate new idea"
        if bs_mode == "mode_a":
            if use_cue_words:
                logger.info("{} using brainstorm_mode_a with cue words.".format(mode_name))
                (
                    message_input, 
                    problem, 
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_with_cue_words(background)
                )
            else:
                logger.info("{} using brainstorm_mode_a without cue words.".format(mode_name))
                (
                    message_input, 
                    problem, 
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_without_cue_words(background)
                )
        elif bs_mode == "mode_b" or bs_mode == "mode_c":
            if use_cue_words:
                logger.info("{} using brainstorm_{} with cue words.".format(mode_name, bs_mode))
                (
                    message_input, 
                    problem, 
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_with_cue_words_bs(background)
                )
            else:
                logger.info("{} using brainstorm_{} without cue words.".format(mode_name, bs_mode))
                (
                    message_input, 
                    problem, 
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_without_cue_words_bs(background)
                )

        idea_modified = self.api_helper.modify_idea(background, idea_filtered)
        median = {
            "problem": problem,
            "initial_idea": idea,
            "filtered_idea": idea_filtered,
        }
        return message_input, idea_modified, median

    def generate_by_inspiration(
        self,
        background: str,
        mode: str,
        bs_mode: str = None,
        use_cue_words: bool = False,
    ):
        mode_name = None
        if mode == "backtracking":
            mode_name = "Backtrack"
        elif mode == "new_idea":
            mode_name = "Generate new idea"
        if bs_mode == "mode_a":
            if use_cue_words:
                logger.info("{} using brainstorm_mode_a with cue words.".format(mode_name))
                (
                    message_input, 
                    problem,
                    inspirations, 
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_with_cue_words_ins(background)
                )
            else:
                logger.info("{} using brainstorm_mode_a without cue words.".format(mode_name))
                (
                    message_input, 
                    problem, 
                    inspirations,
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_without_cue_words_ins(background)
                )
        elif bs_mode == "mode_b" or bs_mode == "mode_c":
            if use_cue_words:
                logger.info("{} using brainstorm_{} with cue words.".format(mode_name, bs_mode))
                (
                    message_input, 
                    problem, 
                    inspirations,
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_with_cue_words_ins_bs(background)
                )
            else:
                logger.info("{} using brainstorm_{} without cue words.".format(mode_name, bs_mode))
                (
                    message_input, 
                    problem, 
                    inspirations,
                    idea, 
                    idea_filtered
                ) = (
                    self.generate_without_cue_words_ins_bs(background)
                )

        idea_modified = self.api_helper.modify_idea(background, idea_filtered)
        median = {
            "problem": problem,
            "inspirations": inspirations,
            "initial_idea": idea,
            "filtered_idea": idea_filtered,
        }
        return message_input, idea_modified, median


@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    default="./configs/datasets.yaml",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--ids-path",
    default="./assets/data/test_acl_2024.json",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-r",
    "--retriever-name",
    default="SNKG",
    type=str,
    required=True,
    help="Retrieve method",
)
@click.option(
    "--brainstorm-mode",
    default="mode_c",
    type=str,
    required=True,
    help="Choose your brainstorm mode (mode_a: no brainstorm, mode_b: brainstorm for idea generation, mode_c: brainstorm for idea generation and retrival)",
)
@click.option(
    "--use-cue-words",
    default=False,
    type=bool,
    required=True,
    help="Use cue words in generation",
)
@click.option(
    "--use-inspiration",
    default=False,
    type=bool,
    required=True,
    help="Use inspiration in generation",
)
@click.option(
    "--num",
    default=100,
    type=int,
    required=False,
    help="The number of papers you want to process",
)
def backtracking(config_path, ids_path, retriever_name, brainstorm_mode, use_cue_words, use_inspiration, num, **kwargs):
    check_env()
    check_embedding() 
    # Configuration
    config = ConfigReader.load(config_path, **kwargs)
    logger.add(
        "log/generate_{}_{}.log".format(time.time(), retriever_name),
        level=config.DEFAULT.log_level,
    )
    logger.info("\nretrieve name : {}".format(retriever_name))
    logger.info("Loaded configuration:\n{}".format(OmegaConf.to_yaml(config)))
    api_helper = APIHelper(config)
    paper_client = PaperClient()
    eval_data = []
    processed_ids = set()
    cur_num = 0
    batch_size = 2
    output_dir = "./assets/output_idea/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_backtracking_{brainstorm_mode}_cue_{use_cue_words}_ins_{use_inspiration}.json")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                eval_data = json.load(f)
                processed_ids = {paper["hash_id"] for paper in eval_data}
                cur_num = len(eval_data)
            except json.JSONDecodeError:
                print("Failed to decode JSON, initializing eval_data as an empty list.")
    print(f"{cur_num} papers have been processed.")
    for line in ids_path:
        # 解析每行的JSON数据
        paper = json.loads(line)
        if paper["hash_id"] in processed_ids:
            print(f"Skipping already processed paper: {paper_id}")
            continue
        logger.info("\nbegin generate paper hash id {}".format(paper["hash_id"]))
        # if "entities" in paper.keys():
        #     entities = paper["entities"]
        # else:
        # 1. 获取背景信息
        paper = paper_client.get_paper_by_id(paper["hash_id"])
        if "motivation" in paper.keys():
            bg = paper["motivation"]
        else:
            print(f"Paper hash_id {paper['hash_id']} doesn't have background...")
            continue
        if brainstorm_mode == "mode_b" or brainstorm_mode == "mode_c":
            brainstorm = api_helper.generate_brainstorm(bg)
        else:
            brainstorm = None
        if "entities" in paper.keys():
            entities = paper["entities"]
        else:
            entities = api_helper.generate_entity_list(bg)
        logger.debug("Original entities from background: {}".format(entities))
        if brainstorm_mode == "mode_c":
            entities_bs = api_helper.generate_entity_list(brainstorm, 10)
            logger.debug("Original entities from brainstorm: {}".format(entities_bs))
            entities_all = list(set(entities)|set(entities_bs))
        else:
            entities_bs = None
            entities_all = entities
        # 2. 获取真实引用文章 (用于评估)
        cite_type = "cite_id_list"
        # cite_type = config.RETRIEVE.cite_type
        if cite_type in paper and len(paper[cite_type]) >= 5:
            target_paper_id_list = paper[cite_type]
        else:
            logger.warning(
                "Hash ID {} cited paper num less than 5...".format(paper["hash_id"])
            )
            continue
        # 3. 检索相关论文
        rt = RetrieverFactory.get_retriever_factory().create_retriever(
            retriever_name, 
            config
        )
        result = rt.retrieve(
            bg, entities_all, need_evaluate=False, target_paper_id_list=[]
        )
        related_paper = result["related_paper"]
        logger.info("Find {} related papers...".format(len(related_paper)))
        entities_rt = result["entities"]
        # 4. 生成IDEA
        if use_cue_words:
            if "contribution" in paper.keys():
                cue_words = api_helper.generate_entity_list(paper["contribution"])
            else:
                print(f"Paper hash_id {paper['hash_id']} doesn't have contribution...")
                cue_words = None
        else:
            cue_words = None
        idea_generator = IdeaGenerator(config, related_paper, cue_words, brainstorm)
        if not use_inspiration:
            message_input, idea_modified, median = idea_generator.generate(
                bg, "backtracking", brainstorm_mode, use_cue_words
            )
        else:
            message_input, idea_modified, median = (
                idea_generator.generate_by_inspiration(
                    bg, "backtracking", brainstorm_mode, use_cue_words
                )
            )
        eval_data.append(
            {
                "hash_id": paper["hash_id"],
                "background": bg,
                "entities_bg": entities,
                "brainstorm" : brainstorm,
                "entities_bs": entities_bs,
                "entities_rt": entities_rt,
                "related_paper": [p["hash_id"] for p in related_paper],
                "input": message_input,
                "cue_words": cue_words,
                "median": median,
                "pred": idea_modified,
                "ground_truth": paper["ground_truth"],
            }
        )
        cur_num += 1
        if cur_num % batch_size == 0:
            with open(
                output_file,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=4)
        if cur_num >= num:
            break
    logger.info("=== Finish ===")
    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)

@main.command()
@click.option(
    "-c",
    "--config-path",
    default="./configs/datasets.yaml",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--ids-path",
    default="./assets/data/test_background.json",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-r",
    "--retriever-name",
    default="SNKG",
    type=str,
    required=True,
    help="Retrieve method",
)
@click.option(
    "--brainstorm-mode",
    default="mode_c",
    type=str,
    required=True,
    help="Choose your brainstorm mode (mode_a: no brainstorm, mode_b: brainstorm for idea generation, mode_c: brainstorm for idea generation and retrival)",
)
@click.option(
    "--use-inspiration",
    default=False,
    type=bool,
    required=True,
    help="Use inspiration in generation",
)
@click.option(
    "--num",
    default=100,
    type=int,
    required=False,
    help="The number of data you want to process",
)
def new_idea(config_path, ids_path, retriever_name, brainstorm_mode, use_inspiration, num, **kwargs):
    check_env()
    check_embedding()
    logger.add(
        "log/generate_{}_{}.log".format(time.time(), retriever_name), level="DEBUG"
    )  # 添加文件输出
    logger.info("Retrieve name: {}".format(retriever_name))
    # Configuration
    config = ConfigReader.load(config_path, **kwargs)
    api_helper = APIHelper(config)
    eval_data = []
    cur_num = 0
    data_num = 0
    batch_size = 2
    bg_ids = set()
    output_dir = "./assets/output_idea/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_new_idea_{brainstorm_mode}_ins_{use_inspiration}.json")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                eval_data = json.load(f)
                bg_ids = {data["background"] for data in eval_data}
                cur_num = len(eval_data)
            except json.JSONDecodeError:
                eval_data = []
    print(f"{cur_num} datas have been processed.")
    for line in ids_path:
        # 解析每行的JSON数据
        data = json.loads(line)
        # 1. 获取背景信息
        if "background" in data.keys():
            bg = data["background"]
        else:
            data_num += 1
            print(f"This data doesn't have background...")
            continue
        if bg in bg_ids:
            data_num += 1
            print(f"Skipping already processed data_{data_num}.")
            continue
        if brainstorm_mode == "mode_b" or brainstorm_mode == "mode_c":
            brainstorm = api_helper.generate_brainstorm(bg)
        else:
            brainstorm = None
        if "cue_words" in data.keys():
            use_cue_words = True
            cue_words = data["cue_words"]
        else:
            use_cue_words = False
            cue_words = None
        entities = api_helper.generate_entity_list(bg)
        logger.debug("Original entities from background: {}".format(entities))
        if brainstorm_mode == "mode_c":
            entities_bs = api_helper.generate_entity_list(brainstorm, 10)
            logger.debug("Original entities from brainstorm: {}".format(entities_bs))
            entities_all = list(set(entities)|set(entities_bs))
        else:
            entities_bs = None
            entities_all = entities
        # 2. 检索相关论文
        rt = RetrieverFactory.get_retriever_factory().create_retriever(
            retriever_name,
            config
        )
        result = rt.retrieve(bg, entities_all, need_evaluate=False, target_paper_id_list=[])
        related_paper = result["related_paper"]
        logger.info("Find {} related papers...".format(len(related_paper)))
        entities_rt = result["entities"]
        # 3. 生成IDEA
        idea_generator = IdeaGenerator(config, related_paper, cue_words, brainstorm)
        if not use_inspiration:
            message_input, idea_modified, median = idea_generator.generate(
                bg, "new_idea", brainstorm_mode, use_cue_words
            )
        else:
            message_input, idea_modified, median = (
                idea_generator.generate_by_inspiration(
                    bg, "new_idea", brainstorm_mode, use_cue_words
                )
            )
        eval_data.append(
            {
                "background": bg,
                "entities_bg": entities,
                "brainstorm" : brainstorm,
                "entities_bs": entities_bs,
                "entities_rt": entities_rt,
                "related_paper": [p["hash_id"] for p in related_paper],
                "input": message_input,
                "cue_words": cue_words,
                "median": median,
                "pred": idea_modified,
            }
        )
        cur_num += 1
        if cur_num % batch_size == 0:
            with open(
                output_file,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=4)
        if cur_num >= num:
            break
    logger.info("=== Finish ===")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
