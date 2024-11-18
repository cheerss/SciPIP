# ai scientist 生成 idea
# Reference: https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/generate_ideas.py
from utils.paper_retriever import RetrieverFactory
from utils.llms_api import APIHelper
from utils.header import ConfigReader
from omegaconf import OmegaConf
import click
import json
from loguru import logger
import warnings
import time
warnings.filterwarnings('ignore')


class AiScientistIdeaGenerator():
    def __init__(self, config) -> None:
        self.api_helper = APIHelper(config)

    def generate(message_input):
        # @LuoYunxiang 
        return

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
    default='../configs/datasets.yaml',
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--ids-path",
    default='../assets/data/test_acl_2024.json',
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-r",
    "--retriever-name",
    default='SNKG',
    type=str,
    required=True,
    help="Retrieve method",
)
@click.option(
    "--llms-api",
    default=None,
    type=str,
    required=False,
    help="The LLMS API alias used. If you do not have separate APIs for summarization and generation, you can use this unified setting. This option is ignored when setting the API to be used by summarization and generation separately",
)
@click.option(
    "--sum-api",
    default=None,
    type=str,
    required=False,
    help="The LLMS API aliases used for summarization. When used, it will invalidate --llms-api",
)
@click.option(
    "--gen-api",
    default=None,
    type=str,
    required=False,
    help="The LLMS API aliases used for generation. When used, it will invalidate --llms-api",
)
def generate(config_path, ids_path, retriever_name, **kwargs):
    logger.add("ai_scientist_generate_{}.log".format(retriever_name), level="DEBUG")
    logger.info("Retrieve name: {}".format(retriever_name))
    # Configuration
    config = ConfigReader.load(config_path, **kwargs)
    api_helper = APIHelper(config)
    eval_data = []
    num = 0
    for line in ids_path:
        # Parse each line's JSON data
        background = json.loads(line)
        bg = background["background"]
        entities = api_helper.generate_entity_list(bg)
        logger.debug("Original entities from background: {}".format(entities))
        rt = RetrieverFactory.get_retriever_factory().create_retriever(
            retriever_name, 
            config
        )
        result = rt.retrieve(bg, entities, need_evaluate=False, target_paper_id_list=[], top_k=5)
        related_paper = result["related_paper"]
        logger.info("Find {} related papers...".format(len(related_paper)))
        title_list = [paper["title"] for paper in related_paper]
        contribution_list = [paper["summary"] for paper in related_paper]
        message_input =  {
          "Name": ",".join(entities),
          "Title": ",".join(title_list),
          "Experiment": ",".join(contribution_list)
        }
        print(message_input)
        exit()
        idea_generator = AiScientistIdeaGenerator(config)
        # idea list(str)
        idea = idea_generator.generate(message_input)
        eval_data.append({
            "background": bg,
            "input": message_input,
            "pred": idea
        })
        num += 1
        if num >= 1:
            break
    logger.info("=== Finish ===")
    with open("ai_scientist_output_new_idea.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
