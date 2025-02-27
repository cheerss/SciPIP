import os
import json
import re
from tqdm import tqdm
import torch
from utils.paper_crawling import PaperCrawling
from utils.paper_client import PaperClient
from utils.hash import generate_hash_id, get_embedding_model
from collections import defaultdict
from utils.header import get_dir, ConfigReader
from utils.llms_api import APIHelper
from utils.paper_retriever import Retriever
from utils import scipdf
import click
from collections import Counter
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

unicode_pattern = r"\u00c0-\u00ff\u0100-\u017f\u0180-\u024f\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff"


def find_methodology(article_dict):
    """For an article dict (representing an article), return the methodology part
    Args:
        article_dict
    Returns:
        methodology: str
    """
    def find_section_index(keywords):
        for i, section in enumerate(article_dict["sections"], 1):
            heading = section["heading"].lower()
            text = section["text"].lower()
            if any(keyword in heading for keyword in keywords):
                return i - 1
            i = -1
        if i == -1:
            for i, section in enumerate(article_dict["sections"], 1):
                heading = section["heading"].lower()
                text = section["text"].lower()
                if any(
                    keyword in re.split(r"(?<=[.!?])\s+", text)[-1]
                    for keyword in keywords
                ):
                    return i
        return -1

    index = find_section_index(["experiment", "evaluation"])
    if index == -1:
        experiments_index = next(
            (
                i
                for i, section in enumerate(article_dict["sections"])
                if "experiment" in section["heading"].lower()
                or "evaluation" in section["heading"].lower()
            ),
            5,
        )
        experiments_index = min(experiments_index, len(article_dict["sections"]))
        texts = [
            section["text"] for section in article_dict["sections"][1:experiments_index]
        ]
        methodology = " ".join(texts)
        return methodology
    texts = [
        section["text"]
        for section in article_dict["sections"][1:index]
        if not any(
            keyword in section["heading"].lower()
            for keyword in ["relate", "previous", "background"]
        )
    ]
    methodology = " ".join(texts)
    return methodology


def count_sb_pairs(text):
    """Find the number of square brackets (possible citations)
    """
    return len(re.findall(r"\[.*?\]", text))


def count_rb_pairs(text):
    """Find the number of round brackets (possible citations)
    """
    return len(re.findall(r"\(.*?\)", text))


def find_cite_paper(introduction, methodology, references):
    """
    Count the number of times []/() appear in the introduction,
    and determine which one is the reference ()/[]
    """
    text = introduction + methodology
    rb_count = count_rb_pairs(introduction)
    sb_count = count_sb_pairs(introduction)
    ## Seems redudant, remove repeated definition of pattern
    # pattern = (
    #     r"\b[A-Z"
    #     + unicode_pattern
    #     + r"][a-zA-Z"
    #     + unicode_pattern
    #     + r"]+(?: and [A-Z"
    #     + unicode_pattern
    #     + r"][a-zA-Z"
    #     + unicode_pattern
    #     + r"]+)?(?: et al\.)?, \d{4}[a-z]?\b"
    # )
    pattern = (
        r"\b[A-Z"
        + unicode_pattern
        + r"][a-zA-Z"
        + unicode_pattern
        + r"]+(?: and [A-Z"
        + unicode_pattern
        + r"][a-zA-Z"
        + unicode_pattern
        + r"]+)?(?: et al\.)?, \d{4}[a-z]?\b"
    )
    temp_list = re.findall(pattern, text)
    ref_list = []
    ref_title = []
    if len(temp_list) > 0:
        pattern = (
            r"\b([A-Z"
            + unicode_pattern
            + r"][a-zA-Z"
            + unicode_pattern
            + r"]+)(?: and [A-Z"
            + unicode_pattern
            + r"][a-zA-Z"
            + unicode_pattern
            + r"]+)?(?: et al\.)?, (\d{4})[a-z]?\b"
        )
        for temp in temp_list:
            match = re.search(pattern, temp)
            ref_list.append({"authors": match.group(1), "year": match.group(2)})
        for i, ref in enumerate(ref_list):
            for j, r in enumerate(references):
                if r["year"] == ref["year"] and ref["authors"] in r["authors"]:
                    ref_title.append(r["title"])
    if len(ref_title) <= 1:
        ref_list = []
        ref_title = []
        if rb_count < sb_count:
            pattern = r"\[\d+(?:,\s*\d+)*\]"
        else:
            pattern = r"\(\d+(?:,\s*\d+)*\)"
        ref_list = re.findall(pattern, text)
        # ref: ['[15, 16]', '[5]', '[2, 3, 8]']
        combined_ref_list = []
        for ref in ref_list:
            numbers = re.findall(r"\d+", ref)
            combined_ref_list.extend(map(int, numbers))
        # Sort
        ref_counts = Counter(combined_ref_list)
        ref_counts = dict(sorted(ref_counts.items()))
        ref_list = list(ref_counts.keys())
        for idx in ref_list:
            if idx < len(references):
                ref_title.append(references[idx]["title"])
    return ref_title


class PaperManager:
    def __init__(self, config, venue_name="acl", year="2013") -> None:
        log_dir = config.DEFAULT.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
        log_file = os.path.join(log_dir, "paper_manager.log")
        logger.add(log_file, level=config.DEFAULT.log_level)
        self.venue_name = venue_name
        self.year = year
        self.data_type = "train"
        self.paper_client = PaperClient()
        self.paper_crawling = PaperCrawling(config, data_type=self.data_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = get_embedding_model(config)
        self.api_helper = APIHelper(config)
        self.retriever = Retriever(config)
        self.paper_id_map = defaultdict()
        self.citemap = defaultdict(set)
        self.year_list = [
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
            "2023",
            "2024",
        ]
        self.config = config
        with open(config.DEFAULT.ignore_paper_id_list, "r", encoding="utf-8") as f:
            try:
                self.ignore_paper_pdf_url = [dic["pdf_url"] for dic in json.load(f)]
            except:
                self.ignore_paper_pdf_url = []

    def create_vector_index(self):
        index_exists = self.paper_client.check_index_exists()
        if not index_exists:
            print("Create vector index paper-embeddings")
            self.paper_client.create_vector_index()

    def clean_entity(self, entity):
        """The extracted entities may be noisy, remove all noisy characters
        Args:
            entity (str): an entity
        Returns:
            cleaned_entity (str): entity after cleaning
        """
        if entity is None:
            return None
        # remove all () and their contents
        cleaned_entity = re.sub(r"\([^)]*\)", "", entity)
        # remove non-word characters (e.g., punctuations)
        cleaned_entity = re.sub(r"[^\w\s]", "", cleaned_entity)
        # replace _ as a whitespace
        cleaned_entity = re.sub(r"_", " ", cleaned_entity)
        # remove multiple continuous blanks, remove leading and trailing spaces 
        # (\s means blank characters)
        cleaned_entity = re.sub(r"\s+", " ", cleaned_entity).strip()
        return cleaned_entity

    def clean_text(self, text):
        return text.replace(", , ", ", ")

    def check_parse(self, paper):
        # Required keys
        required_keys = [
            "abstract",
            "introduction",
            "reference",
            "methodology",
            "reference_filter",
        ]
        # Check for missing keys or None values
        for key in required_keys:
            if key not in paper or paper[key] is None:
                logger.error(
                    f"hash_id: {paper.get('hash_id')} pdf_url: {paper.get('pdf_url')} : "
                    f"Missing or None '{key}' in paper."
                )
                return False
        return True

    def update_paper(
        self,
        paper,
        need_download=False,
        need_parse=False,
        need_summary=False,
        need_get_entities=False,
        need_ground_truth=False,
    ):
        if paper["pdf_url"] in self.ignore_paper_pdf_url:
            logger.warning(
                "hash_id: {}, pdf_url: {} ignore".format(
                    paper["hash_id"], paper["pdf_url"]
                )
            )
            return
        self.paper_client.update_paper_from_client(paper)
        if need_download:
            if not self.paper_crawling.download_paper(paper):
                print(f"download paper {paper['pdf_url']} failed!")
                return
        if need_parse:
            if not self.check_parse(paper):
                logger.debug(f"begin to parse {paper['hash_id']}")
                if not self.paper_crawling.download_paper(paper):
                    logger.error(f"download paper {paper['pdf_url']} failed!")
                    return
                try:
                    article_dict = scipdf.parse_pdf_to_dict(paper["pdf_path"])
                    if "title" not in paper.keys() or paper["title"] is None:
                        paper["title"] = article_dict["title"]
                    paper["abstract"] = article_dict["abstract"]
                    paper["introduction"] = article_dict["sections"][0]["text"]
                    paper["methodology"] = find_methodology(article_dict)
                    reference = []
                    for ref in article_dict["references"]:
                        reference.append(ref["title"])
                    paper["reference"] = reference
                    paper["reference_filter"] = find_cite_paper(
                        paper["introduction"],
                        paper["methodology"],
                        article_dict["references"],
                    )
                    logger.info(f"{paper['hash_id']} parse success")
                except Exception:
                    logger.error(
                        f"{paper['hash_id']}: {paper['pdf_url']}  parse error!"
                    )

        if need_summary:
            if not self.check_parse(paper):
                logger.error(f"paper {paper['hash_id']} need parse first...")
            elif "summary" not in paper.keys():
                result = self.api_helper(
                    paper["title"], paper["abstract"], paper["introduction"]
                )
                if result is not None:
                    paper["summary"] = result["summary"]
                    paper["motivation"] = result["motivation"]
                    paper["contribution"] = result["contribution"]
                    logger.info(f"paper {paper['hash_id']} summary success...")
                else:
                    logger.warning(
                        "hash_id: {}, pdf_url: {} summary failed...".format(
                            paper["hash_id"], paper["pdf_url"]
                        )
                    )
            if need_ground_truth:
                if "ground_truth" not in paper.keys():
                    if (
                        "abstract" in paper.keys()
                        and "contribution" in paper.keys()
                        and "methodology" in paper.keys()
                    ):
                        paper["ground_truth"] = self.api_helper.generate_ground_truth(
                            abstract=paper["abstract"],
                            contribution=paper["contribution"],
                            text=paper["methodology"],
                        )
                        logger.info(f"paper {paper['hash_id']} ground truth success...")
                    else:
                        logger.error("Can't get ground truth...please check")

        # insert paper in database
        if self.check_parse(paper):
            self.paper_client.add_paper_node(paper)
        else:
            return

        if need_get_entities and self.paper_client.check_entity_node_count(
            paper["hash_id"]
        ):
            if (
                paper["abstract"] is None
                or paper["introduction"] is None
                or paper["reference"] is None
            ):
                logger.error(f"paper need parse first")
            entities = self.api_helper.generate_entity_list(paper["abstract"])
            logger.info("hash_id {}, Entities: {}".format(paper["hash_id"], entities))
            if entities is not None:
                self.paper_client.add_entity_node(paper["hash_id"], entities)
            else:
                logger.warning(
                    "hash_id: {}, pdf_url: {} entities None...".format(
                        paper["hash_id"], paper["pdf_url"]
                    )
                )

    def update_paper_local(
        self,
        paper,
        need_download=False,
        need_parse=False,
        need_summary=False,
        need_get_entities=False,
        need_ground_truth=False,
    ):
        """Parse a paper, dump the result into a json file
        """
        if paper["pdf_url"] in self.ignore_paper_pdf_url:
            logger.warning(
                "hash_id: {}, pdf_url: {} ignore".format(
                    paper["hash_id"], paper["pdf_url"]
                )
            )
            return
        # keep the content of the paper node consistent with the database
        self.paper_client.update_paper_from_client(paper)
        if need_download:
            if not self.paper_crawling.download_paper(paper):
                print(f"download paper {paper['pdf_url']} failed!")
                return
        if need_parse:
            if not self.check_parse(paper):  # haven't parse
                logger.debug(f"begin to parse {paper['hash_id']}")
                if not self.paper_crawling.download_paper(paper):
                    logger.error(f"download paper {paper['pdf_url']} failed!")
                    return
                try:
                    article_dict = scipdf.parse_pdf_to_dict(paper["pdf_path"])
                    if "title" not in paper.keys() or paper["title"] is None:
                        paper["title"] = article_dict["title"]
                    paper["abstract"] = article_dict["abstract"]
                    paper["introduction"] = article_dict["sections"][0]["text"]
                    paper["methodology"] = find_methodology(article_dict)
                    reference = []
                    for ref in article_dict["references"]:
                        reference.append(ref["title"])
                    paper["reference"] = reference
                    paper["reference_filter"] = find_cite_paper(
                        paper["introduction"],
                        paper["methodology"],
                        article_dict["references"],
                    )
                    logger.info(f"{paper['hash_id']} parse success")
                except Exception:
                    logger.error(
                        f"{paper['hash_id']}: {paper['pdf_url']}  parse error!"
                    )

        if need_summary:
            if not self.check_parse(paper):
                logger.error(f"paper {paper['hash_id']} need parse first...")
            result = self.api_helper(
                paper["title"], paper["abstract"], paper["introduction"]
            )
            if result is not None:
                paper["summary"] = result["summary"]
                paper["motivation"] = result["motivation"]
                paper["contribution"] = result["contribution"]
                logger.info(f"paper {paper['hash_id']} summary success...")
            else:
                logger.warning(
                    "hash_id: {}, pdf_url: {} summary failed...".format(
                        paper["hash_id"], paper["pdf_url"]
                    )
                )

            if need_ground_truth:
                if (
                    "abstract" in paper.keys()
                    and "contribution" in paper.keys()
                    and "methodology" in paper.keys()
                ):
                    paper["ground_truth"] = self.api_helper.generate_ground_truth(
                        abstract=paper["abstract"],
                        contribution=paper["contribution"],
                        text=paper["methodology"],
                    )
                    logger.info(f"paper {paper['hash_id']} ground truth success...")
                else:
                    logger.error("Can't get ground truth...please check")

        if need_get_entities and self.paper_client.check_entity_node_count(
            paper["hash_id"]
        ):
            if (
                paper["abstract"] is None
                or paper["introduction"] is None
                or paper["reference"] is None
            ):
                logger.error(f"paper need parse first")
            entities = self.api_helper.generate_entity_list(paper["abstract"])
            logger.info("hash_id {}, Entities: {}".format(paper["hash_id"], entities))
            if entities is not None:
                self.paper_client.add_entity_node(paper["hash_id"], entities)
            else:
                logger.warning(
                    "hash_id: {}, pdf_url: {} entities None...".format(
                        paper["hash_id"], paper["pdf_url"]
                    )
                )

        with open(
            self.config.output_path.replace(
                ".json", "_{}.json".format(paper["hash_id"])
            ),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(paper, f)
        return paper

    def update_paper_from_json(
        self,
        need_download=True,
        need_parse=False,
        need_summary=False,
        need_get_entities=False,
        need_ground_truth=False,
    ):
        if self.year != "all":
            logger.info(
                "=== year {}, venue name {} ===".format(self.year, self.venue_name)
            )
            with open(
                f"./assets/paper/{self.venue_name}/{self.venue_name}_{self.year}_paper_list.json",
                "r",
                encoding="utf8",
            ) as f:
                paper_list = json.load(f)
            for paper in tqdm(paper_list):
                self.update_paper(
                    paper,
                    need_download=need_download,
                    need_parse=need_parse,
                    need_summary=need_summary,
                    need_get_entities=need_get_entities,
                    need_ground_truth=need_ground_truth,
                )
        else:
            if self.venue_name == "iccv":
                self.year_list = ["2013", "2015", "2017", "2019", "2021", "2023"]
            elif self.venue_name == "eccv":
                self.year_list = ["2018", "2020", "2022", "2024"]
            for year in self.year_list:
                with open(
                    f"./assets/paper/{self.venue_name}/{self.venue_name}_{year}_paper_list.json",
                    "r",
                    encoding="utf8",
                ) as f:
                    paper_list = json.load(f)
                logger.info(
                    "=== year {}, venue name {} ===".format(year, self.venue_name)
                )
                for paper in tqdm(paper_list):
                    self.update_paper(
                        paper,
                        need_download=need_download,
                        need_parse=need_parse,
                        need_summary=need_summary,
                        need_get_entities=need_get_entities,
                        need_ground_truth=need_ground_truth,
                    )

    def update_paper_from_json_to_json(
        self,
        need_download=True,
        need_parse=False,
        need_summary=False,
        need_get_entities=False,
        need_ground_truth=False,
    ):
        """Parse a paper and dump into the json file
        """
        result = []
        if self.year != "all":
            logger.info(
                "=== year {}, venue name {} ===".format(self.year, self.venue_name)
            )
            with open(
                f"./assets/paper/{self.venue_name}/{self.venue_name}_{self.year}_paper_list.json",
                "r",
                encoding="utf8",
            ) as f:
                paper_list = json.load(f)
            result = [
                self.update_paper_local(
                    paper,
                    need_download=need_download,
                    need_parse=need_parse,
                    need_summary=need_summary,
                    need_get_entities=need_get_entities,
                    need_ground_truth=need_ground_truth,
                )
                for paper in tqdm(paper_list)
            ]

        else:
            if self.venue_name == "iccv":
                self.year_list = ["2013", "2015", "2017", "2019", "2021", "2023"]
            elif self.venue_name == "eccv":
                self.year_list = ["2018", "2020", "2022", "2024"]
            for year in self.year_list:
                with open(
                    f"./assets/paper/{self.venue_name}/{self.venue_name}_{year}_paper_list.json",
                    "r",
                    encoding="utf8",
                ) as f:
                    paper_list = json.load(f)
                logger.info(
                    "=== year {}, venue name {} ===".format(year, self.venue_name)
                )
                subresult = [
                    self.update_paper_local(
                        paper,
                        need_download=need_download,
                        need_parse=need_parse,
                        need_summary=need_summary,
                        need_get_entities=need_get_entities,
                        need_ground_truth=need_ground_truth,
                    )
                    for paper in tqdm(paper_list)
                ]
                result += subresult

        with open(self.config.output_path, "w", encoding="utf8") as f:
            json.dump(result, f)

    def insert_citation(self):
        if self.year != "all":
            year_list = [self.year]
        else:
            year_list = self.year_list
        for year in year_list:
            paper_list = self.paper_client.select_paper(self.venue_name, year)
            for paper in tqdm(paper_list):
                if (
                    self.check_parse(paper)
                    and len(paper["reference"]) > 0
                    and "motivation" in paper.keys()
                    and paper["motivation"] is not None
                ):
                    paper["cite_id_list"] = [
                        generate_hash_id(ref_title)
                        for ref_title in paper["reference_filter"]
                    ]
                    paper["cite_id_list"] = self.paper_client.filter_paper_id_list(
                        paper["cite_id_list"], year=year
                    )
                    paper["all_cite_id_list"] = [
                        generate_hash_id(ref_title) for ref_title in paper["reference"]
                    ]
                    paper["all_cite_id_list"] = self.paper_client.filter_paper_id_list(
                        paper["all_cite_id_list"], year=year
                    )
                    if "entities" not in paper.keys() or len(paper["entities"]) < 3:
                        paper["entities"] = self.api_helper.generate_entity_list(
                            paper["abstract"]
                        )
                        logger.debug(
                            "get entity from context: {}".format(paper["entities"])
                        )
                    logger.debug(
                        "paper hash_id {}, cite_id_list {}, all_cite_id_list {}".format(
                            paper["hash_id"],
                            paper["cite_id_list"],
                            paper["all_cite_id_list"],
                        )
                    )
                else:
                    paper["cite_id_list"] = []
                    paper["all_cite_id_list"] = []
                if (
                    "entities" in paper.keys()
                    and "cite_id_list" in paper.keys()
                    and "all_cite_id_list" in paper.keys()
                ):
                    self.paper_client.add_paper_citation(paper)

    def insert_entity_combinations(self):
        if self.year != "all":
            year_list = [self.year]
        else:
            year_list = self.year_list
        for year in year_list:
            self.paper_client.get_entity_combinations(self.venue_name, year)

    def insert_embedding(self, hash_id=None):
        self.paper_client.add_paper_abstract_embedding(self.embedding_model, hash_id)
        # self.paper_client.add_paper_bg_embedding(self.embedding_model, hash_id)
        # self.paper_client.add_paper_contribution_embedding(
        #     self.embedding_model, hash_id
        # )
        # self.paper_client.add_paper_summary_embedding(self.embedding_model, hash_id)

    def add_new_embedding(self, hash_id=None, to="all"):
        """add new embeddings for abstract, background, contribution, and summary
        """
        postfix_set = {
            "sentence-transformers/all-MiniLM-L6-v2": "",
            "BAAI/llm-embedder": "_llm_embedder",
            "jina/jina-embeddings-v3": "_jina_v3"
        }
        postfix = postfix_set[self.config.DEFAULT.embedding]
        if "jina" in postfix:
            if self.config.DEFAULT.embedding_task == "text-matching":
                postfix += "_text_matching"
            elif self.config.DEFAULT.embedding_task == "retrieval.query":
                postfix += "_query"
            elif self.config.DEFAULT.embedding_task == "retrieval.passage":
                postfix += "_passage"
            else:
                assert False
        if to == "all" or to == "abstract":
            self.paper_client.update_paper_embedding(
                self.embedding_model, hash_id,
                name="abstract", postfix=postfix
            )
        if to == "all" or to == "background":
            self.paper_client.update_paper_embedding(
                self.embedding_model, hash_id,
                name="background", postfix=postfix
            )
        if to == "all" or to == "contribution":
            self.paper_client.update_paper_embedding(
                self.embedding_model, hash_id,
                name="contribution", postfix=postfix
            )
        if to == "all" or to == "summary":
            self.paper_client.update_paper_embedding(
                self.embedding_model, hash_id,
                name="summary", postfix=postfix
            )

    def cosine_similarity_search(self, data_type, context, k=1):
        """
        return related paper: list
        """
        embedding = self.embedding_model.encode(context)
        result = self.paper_client.cosine_similarity_search(data_type, embedding, k)
        return result

    def generate_paper_list(self):
        """Dump paper list into a json file, the json is saved at "folder_path"
        Args:
            None
        Return:
            None
        """
        folder_path = f"./assets/paper/{self.venue_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.year != "all":
            logger.info(
                "=== year {}, venue name {} ===".format(self.year, self.venue_name)
            )
            paper_list = self.paper_crawling.crawling(self.year, self.venue_name)
            with open(
                f"{folder_path}/{self.venue_name}_{self.year}_paper_list.json",
                "w",
            ) as f:
                json.dump(paper_list, f, indent=4, ensure_ascii=False)
        else:
            for year in self.year_list:
                logger.info(
                    "=== year {}, venue name {} ===".format(year, self.venue_name)
                )
                paper_list = self.paper_crawling.crawling(year, self.venue_name)
                with open(
                    f"{folder_path}/{self.venue_name}_{year}_paper_list.json",
                    "w",
                ) as f:
                    json.dump(paper_list, f, indent=4, ensure_ascii=False)


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
    default=get_dir("./configs/datasets.yaml"),
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--year",
    default="2013",
    type=str,
    required=True,
    help="Venue year",
)
@click.option(
    "--venue-name",
    default="acl",
    type=str,
    required=True,
    help="Venue name",
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
def crawling(config_path, year, venue_name, **kwargs):
    """Download paper list in to a json file
    Args:
        config_path (str):
        year (int): the paper's publication data
        venue_name (str): CVPR, etc.
    Resturns:
        None
    """
    # Configuration
    config = ConfigReader.load(config_path, **kwargs)
    pm = PaperManager(config, venue_name, year)
    pm.generate_paper_list()


@main.command()
@click.option(
    "-c",
    "--config-path",
    default=get_dir("./configs/datasets.yaml"),
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--year",
    default="2013",
    type=str,
    required=True,
    help="Venue year",
)
@click.option(
    "--venue-name",
    default="acl",
    type=str,
    required=True,
    help="Venue name",
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
def update(config_path, year, venue_name, **kwargs):
    """Read paper lists from assets/paper directory, insert them into database,
    including downloading, parsing, etc., but not embedding
    """
    # Configuration
    config = ConfigReader.load(config_path, **kwargs)
    pm = PaperManager(config, venue_name, year)
    pm.update_paper_from_json(need_download=True)


@main.command()
@click.option(
    "-c",
    "--config-path",
    default=get_dir("./configs/datasets.yaml"),
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--year",
    default="2013",
    type=str,
    required=True,
    help="Venue year",
)
@click.option(
    "--venue-name",
    default="acl",
    type=str,
    required=True,
    help="Venue name",
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
@click.option(
    "-o",
    "--output",
    default=get_dir("./output/out.json"),
    type=click.File("wb"),
    required=True,
    help="Dataset configuration file in YAML",
)
def local(config_path, year, venue_name, output, **kwargs):
    """Parse papers and dump them into json files
    """
    # Configuration
    output_path = output.name
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    config = ConfigReader.load(config_path, output_path=output_path, **kwargs)
    pm = PaperManager(config, venue_name, year)
    print("###")
    pm.update_paper_from_json_to_json(
        need_download=True, need_parse=True, need_summary=True
    )

@main.command()
@click.option(
    "-c",
    "--config-path",
    default=get_dir("./configs/datasets.yaml"),
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
def embedding(config_path):
    """Insert embedding for papers in the database
    """
    # Configuration
    config = ConfigReader.load(config_path)
    PaperManager(config).insert_embedding()

@main.command()
@click.option(
    "-c",
    "--config-path",
    default=get_dir("./configs/datasets.yaml"),
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
def add_new_embedding(config_path):
    """Insert another new embedding for papers in the database
    """
    # Configuration
    config = ConfigReader.load(config_path)
    PaperManager(config).add_new_embedding(to="all")

@main.command()
@click.option(
    "-c",
    "--config-path",
    default=get_dir("./configs/datasets.yaml"),
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
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
@click.option(
    "--year",
    default="2013",
    type=str,
    required=True,
    help="Venue year",
)
@click.option(
    "--venue-name",
    default="acl",
    type=str,
    required=True,
    help="Venue name",
)
@click.option(
    "-o",
    "--output",
    default=get_dir("./output/out.json"),
    type=click.File("wb"),
    required=True,
    help="Dataset configuration file in YAML",
)
def parse_papers_to_json(config_path, venue_name, year, output, **kwargs):
    """Read json files and download papers, then parse them and dump into jsons
    """
    # Configuration
    output_path = output.name
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    config = ConfigReader.load(config_path, output_path=output_path, **kwargs)
    pm = PaperManager(config, venue_name=venue_name, year=year)
    pm.update_paper_from_json_to_json(
        need_download=True, need_parse=True, need_summary=True
    )


if __name__ == "__main__":
    main()
