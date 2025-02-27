from .api import HelperCompany
import re
import os
from .header import get_dir, Prompt, ConfigReader, get_prompt
import traceback
import openai

TAG_moti = "Motivations:"
TAG_contr = "Details:"


def clean_text(text):
    cleaned_text = re.sub(r"-\s*\n", "", text)
    cleaned_text = re.sub(r"\s*\n\s*", " ", cleaned_text)
    return cleaned_text.strip()


def clean_entities(input_string):
    # 取出括号中的内容
    cleaned_text = re.sub(r"\([^)]*\)", "", input_string)
    # 使用正则表达式删除非字母字符
    cleaned = re.sub(r"[^a-zA-Z\s]", "", input_string)
    # 将多个空格替换为一个空格
    cleaned = re.sub(r"\s+", " ", cleaned)
    # 删除首尾空格
    cleaned = cleaned.strip().lower()
    return cleaned


def get_related_papers_information(paper, index=""):
    return "Related paper {index}: {title}\nSummary: {summary}\nBackgrounds: {motivation}\nContributions: {contribution}\n \n".format(
        index=index,
        title=paper["title"],
        summary=paper["summary"],
        motivation=paper["motivation"],
        contribution=paper["contribution"],
    )


class APIHelper(object):

    def __init__(self, config) -> None:
        super(APIHelper, self).__init__()
        self.config = config
        self.__checkout_config__()
        self.generator = self.get_helper()
        self.prompt = Prompt(get_dir(config.ARTICLE.summarizing_prompt))

    def get_helper(self):
        MODEL_TYPE = os.environ["MODEL_TYPE"]
        MODEL_NAME = os.environ["MODEL_NAME"]
        if MODEL_NAME != "local":
            MODEL_API_KEY = os.environ["MODEL_API_KEY"]
        else:
            MODEL_API_KEY = ""
        BASE_URL = os.environ["BASE_URL"]
        return HelperCompany.get()[MODEL_TYPE](
            MODEL_API_KEY, MODEL_NAME, BASE_URL, timeout=None
        )

    def __checkout_config__(self):
        pass

    def __call__(self, title: str, abstract: str, introduction: str) -> dict:
        # if os.environ["MODEL_NAME"] not in [
        #     "glm4",
        #     "glm4-air",
        #     "qwen-max",
        #     "qwen-plus",
        #     "gpt-4o-mini",
        #     "local",
        # ]:
        #     raise ValueError(f"Check model name...")

        if title is None or abstract is None or introduction is None:
            return None
        try:
            message = [
                self.prompt.queries[0][0](
                    title=title, abstract=abstract, introduction=introduction
                )
            ]
            response1 = self.generator.create(
                messages=message,
            )
            summary = clean_text(response1)
            message.append({"role": "assistant", "content": summary})
            message.append(self.prompt.queries[1][0]())
            response2 = self.generator.create(
                messages=message,
            )
            detail = response2
            motivation = clean_text(detail.split(TAG_moti)[1].split(TAG_contr)[0])
            contribution = clean_text(detail.split(TAG_contr)[1])
            result = {
                "summary": summary,
                "motivation": motivation,
                "contribution": contribution,
            }
        except Exception:
            traceback.print_exc()
            return None
        return result

    def generate_concise_method(self, methodology: str):
        prompt = get_prompt()
        if methodology is None:
            return None
        try:
            message = [
                prompt[0][0](),
                prompt[1][0](
                    methodology=methodology
                ),
            ]
            detail_method = self.generator.create(
                messages=message,
            )
        except Exception:
            traceback.print_exc()
            return None
        return detail_method

    def generate_entity_list(self, abstract: str, max_num: int = 5) -> list:
        prompt = get_prompt()

        if abstract is None:
            return None
        try:
            examples_str = "\n".join(
                f"[content]: {example['content']}\n[entity]: {example['entities']}\n###\n"
                for example in prompt[1][0].data
            )
            message = [
                prompt[0][0](),
                prompt[1][0](
                    examples=examples_str, content=abstract, max_num=str(max_num)
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            entities = response
            entity_list = entities.strip().split(", ")
            clean_entity_list = []
            for entity in entity_list:
                entity = clean_entities(entity)
                if len(entity.split()) <= 20:
                    clean_entity_list.append(entity)

            if "entity" not in abstract.lower() and "entities" not in abstract.lower():
                clean_entity_list = [
                    re.sub(
                        r"\bentity\b|\bentities\b", "", e, flags=re.IGNORECASE
                    ).strip()
                    for e in clean_entity_list
                ]
                clean_entity_list = [e for e in clean_entity_list if e]
                clean_entity_list = [clean_entities(e) for e in clean_entity_list]
        except Exception:
            traceback.print_exc()
            return None
        return clean_entity_list

    def generate_brainstorm(self, background: str) -> str:
        prompt = get_prompt()

        if background is None:
            print("Input background is empty ...")
            return None
        try:
            # Initial brainstorming to generate raw ideas
            message = [prompt[0][0](), prompt[1][0](background=background)]
            # Call the API to generate brainstorming ideas
            response_brainstorming = self.generator.create(
                messages=message,
            )
            brainstorming_ideas = response_brainstorming

        except Exception:
            traceback.print_exc()
            return None

        return brainstorming_ideas
    
    def expand_idea(self, background: str, idea: str) -> str:
        prompt = get_prompt()

        if background is None:
            print("Input background is empty ...")
            return None
        try:
            # Initial brainstorming to generate raw ideas
            message = [prompt[0][0](), prompt[1][0](background=background, brief_idea=idea)]
            # Call the API to generate brainstorming ideas
            detail_ideas = self.generator.create(
                messages=message,
            )

        except Exception:
            traceback.print_exc()
            return None

        return detail_ideas

    def expand_background(self, brief_background: str, keywords: str) -> str:
        prompt = get_prompt()

        if brief_background is None:
            print("Input brief background is empty ...")
            return None
        try:
            # Initial brainstorming to generate raw ideas
            message = [prompt[0][0](), prompt[1][0](brief_background=brief_background, keywords=keywords)]
            # Call the API to generate brainstorming ideas
            detail_background= self.generator.create(
                messages=message,
            )

        except Exception:
            traceback.print_exc()
            return None

        return detail_background

    def generate_problem(self, background: str, related_papers: list[dict]):
        prompt = get_prompt()
        if background is None or related_papers is None:
            return None
        try:
            related_papers_information = "".join(
                [
                    get_related_papers_information(paper, i + 1)
                    for i, paper in enumerate(related_papers)
                ]
            )
            message_input = prompt[1][0](
                background=background,
                related_papers_information=related_papers_information,
            )
            message = [prompt[0][0](), message_input]
            response = self.generator.create(
                messages=message,
            )
            problem = response
        except Exception:
            traceback.print_exc()
            return None
        return problem, message_input

    def generate_problem_with_cue_words(
        self, background: str, related_papers: list[dict], cue_words: list
    ):
        prompt = get_prompt()

        if background is None or related_papers is None or cue_words is None:
            return None
        try:
            related_papers_information = "".join(
                [
                    get_related_papers_information(paper, i + 1)
                    for i, paper in enumerate(related_papers)
                ]
            )
            message_input = prompt[1][0](
                background=background,
                related_papers_information=related_papers_information,
                cue_words=cue_words,
            )
            message = [prompt[0][0](), message_input]
            response = self.generator.create(
                messages=message,
            )
            problem = response
        except Exception:
            traceback.print_exc()
            return None
        return problem, message_input

    def generate_inspiration(self, problem: str, related_paper: dict):
        prompt = get_prompt()
        if problem is None or related_paper is None:
            return None
        try:
            related_paper_information = get_related_papers_information(related_paper)
            message = [
                prompt[0][0](),
                prompt[1][0](
                    problem=problem, related_paper_information=related_paper_information
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            inspiration = response
        except Exception:
            traceback.print_exc()
            return None
        return inspiration
    

    def generate_inspiration_with_detail_method(self, background: str, detail_method: str):
        prompt = get_prompt()
        if background is None or detail_method is None:
            return None
        try:
            message = [
                prompt[0][0](),
                prompt[1][0](
                    background=background, detail_method=detail_method
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            inspiration = response
        except Exception:
            traceback.print_exc()
            return None
        return inspiration

    def generate_inspiration_with_cue_words(
        self, problem: str, related_paper: dict, cue_words: list
    ):
        prompt = get_prompt()

        if problem is None or related_paper is None or cue_words is None:
            return None
        try:
            related_paper_information = get_related_papers_information(related_paper)
            message = [
                prompt[0][0](),
                prompt[1][0](
                    problem=problem,
                    related_paper_information=related_paper_information,
                    cue_words=cue_words,
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            inspiration = response
        except Exception:
            traceback.print_exc()
            return None
        return inspiration

    def generate_idea(self, problem: str, related_papers: list[dict]) -> str:
        prompt = get_prompt()

        if problem is None or related_papers is None:
            return None
        try:
            related_papers_information = "".join(
                [
                    get_related_papers_information(paper, i + 1)
                    for i, paper in enumerate(related_papers)
                ]
            )
            message = [
                prompt[0][0](),
                prompt[1][0](
                    problem=problem,
                    related_papers_information=related_papers_information,
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            idea = response
        except Exception:
            traceback.print_exc()
            return None
        return idea

    def generate_idea_with_cue_words(
        self, problem: str, related_papers: list[dict], cue_words: list
    ) -> str:
        prompt = get_prompt()

        if problem is None or related_papers is None or cue_words is None:
            return None
        try:
            related_papers_information = "".join(
                [
                    get_related_papers_information(paper, i + 1)
                    for i, paper in enumerate(related_papers)
                ]
            )
            message = [
                prompt[0][0](),
                prompt[1][0](
                    problem=problem,
                    related_papers_information=related_papers_information,
                    cue_words=cue_words,
                ),
            ]

            response = self.generator.create(
                messages=message,
            )
            idea = response
        except Exception:
            traceback.print_exc()
            return None
        return idea

    def generate_idea_by_inspiration(self, background: str, inspirations: list[str]):
        prompt = get_prompt()

        if background is None or inspirations is None:
            return None
        try:
            inspirations_text = "".join(
                [
                    "Inspiration {i}: ".format(i=i + 1) + "\n" + inspiration + "\n \n"
                    for i, inspiration in enumerate(inspirations)
                ]
            )

            message = [
                prompt[0][0](),
                prompt[1][0](background=background, inspirations=inspirations_text),
            ]
            response = self.generator.create(
                messages=message,
            )
            idea = response
        except Exception:
            traceback.print_exc()
            return None
        return idea

    def generate_idea_by_inspiration_with_cue_words(
        self, problem: str, inspirations: list[str], cue_words: list
    ):
        prompt = get_prompt()

        if problem is None or inspirations is None or cue_words is None:
            return None
        try:
            inspirations_text = "".join(
                [
                    "Inspiration {i}: ".format(i=i + 1) + "\n" + inspiration + "\n \n"
                    for i, inspiration in enumerate(inspirations)
                ]
            )

            message = [
                prompt[0][0](),
                prompt[1][0](
                    problem=problem,
                    inspirations_text=inspirations_text,
                    cue_words=cue_words,
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            idea = response
        except Exception:
            traceback.print_exc()
            return None
        return idea

    def integrate_idea(self, background: str, brainstorm: str, idea: str) -> str:
        prompt = get_prompt()

        if background is None or brainstorm is None or idea is None:
            return None
        try:
            message = [
                prompt[0][0](),
                prompt[1][0](background=background, brainstorm=brainstorm, idea=idea),
            ]
            response = self.generator.create(
                messages=message,
            )
            idea = response
        except Exception:
            traceback.print_exc()
            return None
        return idea

    def filter_idea(self, idea: str, background: str) -> str:
        prompt = get_prompt()

        if background is None or idea is None:
            return None
        try:
            message = [
                prompt[0][0](),
                prompt[1][0](
                    idea=idea,
                    background=background,
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            idea_filtered = response
        except Exception:
            traceback.print_exc()
            return None
        return idea_filtered

    def modify_idea(self, background: str, idea: str) -> str:
        prompt = get_prompt()

        if background is None or idea is None:
            return None
        try:
            message = [
                prompt[0][0](),
                prompt[1][0](
                    background=background,
                    idea=idea,
                ),
            ]
            response = self.generator.create(
                messages=message,
            )
            idea_modified = response
        except Exception:
            traceback.print_exc()
            return None
        return idea_modified

    def generate_ground_truth(self, abstract: str, contribution: str, text: str) -> str:
        prompt = get_prompt()

        ground_truth = None
        if abstract is None or contribution is None or text is None:
            return None
        try:
            message = [
                prompt[0][0](abstract=abstract, contribution=contribution, text=text)
            ]
            response = self.generator.create(
                messages=message,
            )
            ground_truth = response
        except Exception:
            traceback.print_exc()
        return ground_truth

    def transfer_form(self, idea: str):
        prompt = get_prompt()

        if idea is None:
            return None
        try:
            message = [prompt[0][0](idea=idea)]
            response = self.generator.create(
                messages=message,
            )
            idea_norm = response
        except Exception:
            traceback.print_exc()
            return None
        return idea_norm

    def select_contribution(self, idea: str, contribution: list[str]) -> str:
        prompt = get_prompt()

        if idea is None or contribution is None:
            return None
        try:
            reference_ideas = "".join(
                [
                    "Idea {i}: ".format(i=i + 1) + "\n" + idea + "\n \n"
                    for i, idea in enumerate(contribution)
                ]
            )
            message = [prompt[0][0](idea=idea, reference_ideas=reference_ideas)]
            response = self.generator.create(
                messages=message,
                max_tokens=10,
            )
            index = response
        except Exception:
            traceback.print_exc()
            return None
        return index

    def get_similarity_score(self, idea: str, contribution: str) -> str:
        prompt = get_prompt()

        if idea is None or contribution is None:
            return None
        try:
            message = [prompt[0][0](idea=idea, reference_idea=contribution)]
            response = self.generator.create(
                messages=message,
                max_tokens=10,
            )
            score = response
        except Exception:
            traceback.print_exc()
            return None
        return score

    def novelty_eval(
        self,
        current_round: int,
        num_rounds: int,  # TODO unused var
        max_num_iterations: int,
        idea: str,
        last_query_results: str,
        msg_history: list,
    ):
        prompt = get_prompt()

        if msg_history is None:
            msg_history = []
        try:
            new_msg_history = msg_history + [
                prompt[1][0](
                    current_round=current_round,
                    num_rounds=max_num_iterations,
                    idea=idea,
                    last_query_results=last_query_results,
                )
            ]
            response = self.generator.create(
                messages=[
                    prompt[0][0](num_rounds=max_num_iterations),
                    *new_msg_history,
                ],
                temperature=0.75,
                max_tokens=3000,
                n=1,
                stop=None,
                seed=0,
            )
            content = response
            new_msg_history = new_msg_history + [
                {"role": "assistant", "content": content}
            ]

        except Exception:
            traceback.print_exc()
            return None
        return content, new_msg_history

    def compare_same(
        self, idea1: str, idea2: str, idea3: str, idea4: str, idea5: str
    ) -> str:
        prompt = get_prompt()

        if not all([idea1, idea2, idea3, idea4, idea5]):
            return None
        try:
            message = (
                [
                    prompt[0][0](),
                    prompt[0][0](
                        idea1=idea1, idea2=idea2, idea3=idea3, idea4=idea4, idea5=idea5
                    ),
                ],
            )
            response = self.generator.create(
                messages=message,
            )
            result = response
        except Exception:
            traceback.print_exc()
            return None
        return result

    def compare_all(self, idea1: str, idea2: str) -> str:
        prompt = get_prompt()

        if idea1 is None or idea2 is None:
            return None
        try:
            message = (
                [
                    prompt[0][0](),
                    prompt[0][0](
                        idea1=idea1,
                        idea2=idea2,
                    ),
                ],
            )
            response = self.generator.create(
                messages=message,
            )
            result = response
        except Exception:
            traceback.print_exc()
            return None
        return result

    def compare_novelty_and_feasibility(self, idea1: str, idea2: str) -> str:
        prompt = get_prompt()

        if idea1 is None or idea2 is None:
            return None
        try:
            message = (
                [
                    prompt[0][0](),
                    prompt[0][0](
                        idea1=idea1,
                        idea2=idea2,
                    ),
                ],
            )
            response = self.generator.create(
                messages=message,
            )
            result = response
        except Exception:
            traceback.print_exc()
            return None
        return result

    def compare_novelty(self, idea1: str, idea2: str) -> str:
        prompt = get_prompt()

        if idea1 is None or idea2 is None:
            return None
        try:
            message = (
                [
                    prompt[0][0](),
                    prompt[0][0](
                        idea1=idea1,
                        idea2=idea2,
                    ),
                ],
            )
            response = self.generator.create(
                messages=message,
            )
            result = response
        except Exception:
            traceback.print_exc()
            return None
        return result

    def compare_feasibility(self, idea1: str, idea2: str) -> str:
        prompt = get_prompt()

        if idea1 is None or idea2 is None:
            return None
        try:
            message = (
                [
                    prompt[0][0](),
                    prompt[0][0](
                        idea1=idea1,
                        idea2=idea2,
                    ),
                ],
            )
            response = self.generator.create(
                messages=message,
            )
            result = response
        except Exception:
            traceback.print_exc()
            return None
        return result


if __name__ == "__main__":
    config = ConfigReader.load("/mnt/llms/data/scimon-plus-data/configs/datasets.yaml")
    api_helper = APIHelper(config=config)
