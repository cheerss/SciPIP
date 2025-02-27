import json
from utils.paper_retriever import RetrieverFactory
from utils.llms_api import APIHelper
from utils.header import ConfigReader
from utils.hash import check_env, check_embedding
from generator import IdeaGenerator
import functools


class Backend(object):
    def __init__(self) -> None:
        CONFIG_PATH = "./configs/datasets.yaml"
        EXAMPLE_PATH = "./assets/data/example.json"
        USE_INSPIRATION = True
        BRAINSTORM_MODE = "mode_c"

        self.config = ConfigReader.load(CONFIG_PATH)
        check_env()
        check_embedding(self.config.DEFAULT.embedding)
        RETRIEVER_NAME = self.config.RETRIEVE.retriever_name
        self.api_helper = APIHelper(self.config)
        self.retriever_factory = (
            RetrieverFactory.get_retriever_factory().create_retriever(
                RETRIEVER_NAME, self.config
            )
        )
        self.idea_generator = IdeaGenerator(self.config, None)
        self.use_inspiration = USE_INSPIRATION
        self.brainstorm_mode = BRAINSTORM_MODE
        self.examples = self.load_examples(EXAMPLE_PATH)

    def load_examples(self, path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading examples from {path}: {e}")
            return []
        
    def background2entities_callback(self, background):
        return self.api_helper.generate_entity_list(background)

    def background2expandedbackground_callback(self, background, entities):
        keywords_str = functools.reduce(lambda x, y: f"{x}, {y}", entities)
        expanded_background = self.api_helper.expand_background(background, keywords_str)
        return expanded_background

    def background2brainstorm_callback(self, expanded_background):
        return self.api_helper.generate_brainstorm(expanded_background)

    def brainstorm2entities_callback(self, brainstorm, entities):
        entities_bs = self.api_helper.generate_entity_list(brainstorm, 10)
        entities_all = list(set(entities) | set(entities_bs))
        return entities_all

    def upload_json_callback(self, input):
        with open(input, "r") as json_file:
            contents = json_file.read()
            json_contents = json.loads(contents)
        return [json_contents["background"], contents]

    def entities2literature_callback(self, expanded_background, entities):
        result = self.retriever_factory.retrieve(
            expanded_background, entities, need_evaluate=False, target_paper_id_list=[]
        )
        res = []
        for i, p in enumerate(result["related_paper"]):
            res.append(f'{p["title"]}. {p["venue_name"].upper()} {p["year"]}.')
        return res, result["related_paper"]

    def literature2initial_ideas_callback(
        self, expanded_background, brainstorms, retrieved_literature
    ):
        self.idea_generator.paper_list = retrieved_literature
        self.idea_generator.brainstorm = brainstorms
        _, _, inspirations, initial_ideas, idea_filtered, final_ideas = (
            self.idea_generator.generate_ins_bs(expanded_background)
        )
        return idea_filtered, final_ideas

    def initial2final_callback(self, initial_ideas, final_ideas):
        return final_ideas

    def get_demo_i(self, i):
        if 0 <= i < len(self.examples):
            return self.examples[i].get("background", "Background not found.")
        else:
            return "Example not found. Please select a valid index."
