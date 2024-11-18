import json
from utils.paper_retriever import RetrieverFactory
from utils.llms_api import APIHelper
from utils.header import ConfigReader
from generator import IdeaGenerator

class Backend(object):
    def __init__(self) -> None:
        CONFIG_PATH = "./configs/datasets.yaml"
        EXAMPLE_PATH = "./assets/data/example.json"
        USE_INSPIRATION = True
        BRAINSTORM_MODE = "mode_c"

        self.config = ConfigReader.load(CONFIG_PATH)
        RETRIEVER_NAME = self.config.RETRIEVE.retriever_name
        self.api_helper = APIHelper(self.config)
        self.retriever_factory = RetrieverFactory.get_retriever_factory().create_retriever(
            RETRIEVER_NAME,
            self.config
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

    def background2brainstorm_callback(self, background, json_strs=None):
        if json_strs is not None: # only for DEBUG_MODE
            json_contents = json.loads(json_strs)
            return json_contents["brainstorm"]
        else:
            return self.api_helper.generate_brainstorm(background)

    def brainstorm2entities_callback(self, background, brainstorm, json_strs=None):
        if json_strs is not None: # only for DEBUG_MODE
            json_contents = json.loads(json_strs)
            entities_bg = json_contents["entities_bg"]
            entities_bs = json_contents["entities_bs"]
            entities_all = entities_bg + entities_bs
            # return gr.CheckboxGroup(choices=entities, value=entities, label="Expanded key words", visible=True)
            return entities_all
        else:
            entities_bg = self.api_helper.generate_entity_list(background)
            entities_bs = self.api_helper.generate_entity_list(brainstorm, 10)
            entities_all = list(set(entities_bg) | set(entities_bs))
            # return extracted_entities
            # return gr.CheckboxGroup(choices=entities_all, value=entities_all, label="Expanded key words", visible=True)
            return entities_all

    def upload_json_callback(self, input):
        # print(type(input))
        # print(len(input))
        # print(input) # temp file path
        with open(input, "r") as json_file:
            contents = json_file.read()
            json_contents = json.loads(contents)
        return [json_contents["background"], contents]

    def entities2literature_callback(self, background, entities, json_strs=None):
        if json_strs is not None:
            result = json.loads(json_strs)
            res = []
            for i, p in enumerate(result["related_paper"]):
                res.append(str(p))
        else:
            result = self.retriever_factory.retrieve(background, entities, need_evaluate=False, target_paper_id_list=[])
            res = []
            for i, p in enumerate(result["related_paper"]):
                res.append(f'{p["title"]}. {p["venue_name"].upper()} {p["year"]}.')
        return res, result["related_paper"]

    def literature2initial_ideas_callback(self, background, brainstorms, retrieved_literature, json_strs=None):
        if json_strs is not None:
            json_contents = json.loads(json_strs)
            return json_contents["median"]["initial_idea"]
        else:
            self.idea_generator.paper_list = retrieved_literature
            self.idea_generator.brainstorm = brainstorms
            if self.use_inspiration:
                message_input, idea_modified, median = (
                self.idea_generator.generate_by_inspiration(
                    background, "new_idea", self.brainstorm_mode, False)
                )
            else:
                message_input, idea_modified, median = self.idea_generator.generate(
                    background, "new_idea", self.brainstorm_mode, False
                )
            return median["initial_idea"], idea_modified
        
    def initial2final_callback(self, initial_ideas, final_ideas, json_strs=None):
        if json_strs is not None:
            json_contents = json.loads(json_strs)
            return json_contents["median"]["modified_idea"]
        else:
            return final_ideas

    def get_demo_i(self, i):
        if 0 <= i < len(self.examples):
            return self.examples[i].get("background", "Background not found.")
        else:
            return "Example not found. Please select a valid index."
    #     return ("The application scope of large-scale language models such as GPT-4 and LLaMA "
    # "has rapidly expanded, demonstrating powerful capabilities in natural language processing "
    # "and multimodal tasks. However, as the size and complexity of the models increase, understanding "
    # "how they make decisions becomes increasingly difficult. Challenge: 1 The complexity of model "
    # "interpretation: The billions of parameters and nonlinear decision paths within large-scale language "
    # "models make it very difficult to track and interpret specific outputs. The existing interpretation "
    # "methods usually only provide a local perspective and are difficult to systematize. 2. Transparency "
    # "and Fairness: In specific scenarios, models may exhibit biased or discriminatory behavior. Ensuring "
    # "the transparency of these models, reducing bias, and providing credible explanations is one of the current challenges.")
