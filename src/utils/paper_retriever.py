import torch
import itertools
import threading
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from loguru import logger
from abc import ABCMeta, abstractmethod
from .paper_client import PaperClient
from .paper_crawling import PaperCrawling
from .llms_api import APIHelper
from .hash import get_embedding_model


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


def can_merge(uf, similarity_matrix, i, j, threshold):
    """Condition of i and j can be merged: After merging, the similarity of any two nodes 
    from root_i and root_j are larger than threshold
    """
    root_i = uf.find(i)
    root_j = uf.find(j)
    for k in range(len(similarity_matrix)):
        if uf.find(k) == root_i or uf.find(k) == root_j:
            if (
                similarity_matrix[i][k] < threshold
                or similarity_matrix[j][k] < threshold
            ):
                return False
    return True


class CoCite:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CoCite, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            logger.debug("init co-cite map begin...")
            self.paper_client = PaperClient()
            citemap = self.paper_client.build_citemap()
            self.comap = defaultdict(lambda: defaultdict(int))
            for paper_id, cited_id in citemap.items():
                for id0, id1 in itertools.combinations(cited_id, 2):
                    # ensure comap[id0][id1] == comap[id1][id0]
                    self.comap[id0][id1] += 1
                    self.comap[id1][id0] += 1
            logger.debug("init co-cite map success")
            CoCite._initialized = True

    def get_cocite_ids(self, id_, k=1):
        """
        """
        sorted_items = sorted(self.comap[id_].items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_items[:k]
        paper_ids = []
        for item in top_k:
            paper_ids.append(item[0])
        paper_ids = self.paper_client.filter_paper_id_list(paper_ids)
        return paper_ids


class Retriever(object):
    """The superclass of all retrievers
    Args:
        config: 
    Returns:
        A Retriever instance
    """
    __metaclass__ = ABCMeta
    retriever_name = "BASE"

    def __init__(self, config):
        self.config = config
        self.use_cocite = config.RETRIEVE.use_cocite
        self.use_cluster_to_filter = config.RETRIEVE.use_cluster_to_filter
        self.paper_client = PaperClient()
        self.cocite = CoCite()
        self.api_helper = APIHelper(config=config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = get_embedding_model(config)
        self.paper_crawling = PaperCrawling(config=config)
        if self.config.DEFAULT.embedding == "sentence-transformers/all-MiniLM-L6-v2":
            self.embedding_postfix = ""
        elif self.config.DEFAULT.embedding == "BAAI/llm-embedder":
            self.embedding_postfix = "_llm_embedder"
        elif self.config.DEFAULT.embedding == "jinaai/jina-embeddings-v3":
            self.embedding_postfix = "_jina_v3"
            if self.config.DEFAULT.embedding_database == "text-matching":
                self.embedding_postfix += "_text_matching"
            elif self.config.DEFAULT.embedding_database == "retrieval.query":
                self.embedding_postfix += "_query"
            elif self.config.DEFAULT.embedding_database == "retrieval.passage":
                self.embedding_postfix += "_passage"
    @abstractmethod
    def retrieve(self, bg, entities, use_evaluate):
        """Retrieve papers, should be implemented by the sub-class
        Args:
            None
        Returns:
            None
        """
        pass

    def retrieve_entities_by_enties(self, entities):
        """The method do three things:
        1. Expand entities according to entities co-occurence
        2. Count the number of papers related to each expanded entity. Sort entities in terms of their occurence times in ascending order
        3. Initial new entities. Retrieve entities one by one until the number of related papers reach a threshold
        Args:
            entities: A List of entities, e.g., [str, str, ...]
        Returns:
            new_entities: A List of entities after expansion, e.g., [str, str, ...]
        """
        # TODO: KG
        expand_entities = self.paper_client.find_related_entities_by_entity_list(
            entities,
            n=self.config.RETRIEVE.kg_jump_num,
            k=self.config.RETRIEVE.kg_cover_num,
            relation_name=self.config.RETRIEVE.relation_name,
        )
        expand_entities = list(set(entities + expand_entities))
        entity_paper_num_dict = self.paper_client.get_entities_related_paper_num(
            expand_entities
        )
        new_entities = []
        entity_paper_num_dict = {
            k: v for k, v in entity_paper_num_dict.items() if v != 0
        }
        entity_paper_num_dict = dict(
            sorted(entity_paper_num_dict.items(), key=lambda item: item[1])
        )
        sum_paper_num = 0
        for key, value in entity_paper_num_dict.items():
            if sum_paper_num <= self.config.RETRIEVE.sum_paper_num:
                sum_paper_num += value
                new_entities.append(key)
            elif (
                value < self.config.RETRIEVE.limit_num
                and sum_paper_num < self.config.RETRIEVE.sum_paper_num
            ):
                sum_paper_num += value
                new_entities.append(key)
        return new_entities

    def update_related_paper(self, paper_id_list):
        """
        Args:
            paper_id_list (List of hash_id): e.g., [1231214, 46345]
        Return:
            related_paper (List of dict):
        """
        related_paper = self.paper_client.update_papers_from_client(paper_id_list)
        return related_paper

    def calculate_similarity(self, entities, related_entities_list, use_weight=False):
        """[Deprecated] Calculate the similarities between two lists of entities
        """
        if use_weight:
            vec1 = self.vectorizer.transform([" ".join(entities)]).toarray()[0]
            weighted_vec1 = np.array(
                [
                    vec1[i] * self.log_inverse_freq.get(word, 1)
                    for i, word in enumerate(self.vectorizer.get_feature_names_out())
                ]
            )
            vecs2 = self.vectorizer.transform(
                [
                    " ".join(related_entities)
                    for related_entities in related_entities_list
                ]
            ).toarray()
            weighted_vecs2 = np.array(
                [
                    [
                        vec2[i] * self.log_inverse_freq.get(word, 1)
                        for i, word in enumerate(
                            self.vectorizer.get_feature_names_out()
                        )
                    ]
                    for vec2 in vecs2
                ]
            )
            similarity = cosine_similarity([weighted_vec1], weighted_vecs2)[0]
        else:
            vec1 = self.vectorizer.transform([" ".join(entities)])
            vecs2 = self.vectorizer.transform(
                [
                    " ".join(related_entities)
                    for related_entities in related_entities_list
                ]
            )
            similarity = cosine_similarity(vec1, vecs2)[0]
        return similarity

    def cal_related_score(
        self, embedding, related_paper_id_list, type_name="background_embedding"
    ):
        """Calculate the cosine similarity between the input background's embedding and
        given list of papers
        Args:
            embedding: the embedding of the input background
            related_paper_id_list (List of int): the paper ids in the database
        Returns:
            Empty dict: {}
            Empty dict: {}
            score_all_dict: 
                paper_id1: score1,
                paper_id2: score2,
                ...
        """
        score_1 = np.zeros((len(related_paper_id_list)))
        # score_2 = np.zeros((len(related_paper_id_list)))
        origin_vector = torch.tensor(embedding).to(self.device).unsqueeze(0)
        context_embeddings = self.paper_client.get_papers_attribute(
            related_paper_id_list, type_name
        )
        if len(context_embeddings) > 0:
            context_embeddings = torch.tensor(context_embeddings).to(self.device)
            score_1 = torch.nn.functional.cosine_similarity(
                origin_vector, context_embeddings
            )
            score_1 = score_1.cpu().numpy()
            if self.config.RETRIEVE.need_normalize:
                score_1 = score_1 / np.max(score_1)
        score_all_dict = dict(zip(related_paper_id_list, score_1))
        # score_en_dict = dict(zip(related_paper_id_list, score_2))
        """
        score_all_dict = dict(
            zip(
                related_paper_id_list,
                score_1 * self.config.RETRIEVE.alpha
                + score_2 * self.config.RETRIEVE.beta,
            )
        )
        """
        return {}, {}, score_all_dict

    def filter_related_paper(self, score_dict, top_k):
        """Pick top_k papers from all retrieved papers in terms of score_dict. If clustering
        is not used, top_k papers with highest scores will be picked. If clustering is used,
        we will pick papers from each cluster in turn util top_k papers are chosen.
        Args:
            score_dict (dict): dict of (paper_id, similarity with user input background)
            top_k (int): pick top_k papers
        Returns:

        """
        if len(score_dict) <= top_k:
            return list(score_dict.keys())
        if not self.use_cluster_to_filter:
            paper_id_list = (
                list(score_dict.keys())[:top_k]
                if len(score_dict) >= top_k
                else list(score_dict.keys())
            )
            return paper_id_list
        else:
            ## Calculate the final embedding for each paper, which is the weighted average
            ## background_embedding (embedding), contribution_embedding, and summary_embedding.
            # clustering filter, ensure that each category the highest score save first
            # background embedding
            paper_id_list = list(score_dict.keys())
            paper_embedding_list = [
                self.paper_client.get_paper_attribute(paper_id, f"background_embedding{self.embedding_postfix}")
                for paper_id in paper_id_list
            ]
            paper_embedding = np.array(paper_embedding_list)
            # contribution embedding
            paper_embedding_list = [
                self.paper_client.get_paper_attribute(
                    paper_id, f"contribution_embedding{self.embedding_postfix}"
                )
                for paper_id in paper_id_list
            ]
            paper_contribution_embedding = np.array(paper_embedding_list)
            # summary embedding
            paper_embedding_list = [
                self.paper_client.get_paper_attribute(paper_id, f"summary_embedding{self.embedding_postfix}")
                for paper_id in paper_id_list
            ]
            paper_summary_embedding = np.array(paper_embedding_list)
            # abstract embedding
            paper_embedding_list = [
                self.paper_client.get_paper_attribute(paper_id, f"abstract_embedding{self.embedding_postfix}")
                for paper_id in paper_id_list
            ]
            paper_abstract_embedding = np.array(paper_embedding_list)

            weight_background = self.config.RETRIEVE.s_bg
            weight_contribution = self.config.RETRIEVE.s_contribution
            weight_summary = self.config.RETRIEVE.s_summary
            weight_abstract = self.config.RETRIEVE.s_abstract
            paper_embedding = (
                weight_background * paper_embedding
                + weight_contribution * paper_contribution_embedding
                + weight_summary * paper_summary_embedding
                + weight_abstract * paper_abstract_embedding
            )

            ## similarity_matrix of all retrieved papers
            similarity_matrix = np.dot(paper_embedding, paper_embedding.T)
            related_labels = self.cluster_algorithm(paper_id_list, similarity_matrix)
            related_paper_label_dict = dict(zip(paper_id_list, related_labels))
            label_group = {}
            for paper_id, label in related_paper_label_dict.items():
                if label not in label_group:
                    label_group[label] = []
                label_group[label].append(paper_id)
            paper_id_list = []
            # randomly pick a paper from each cluster in turn until top_k papers are chosen
            while len(paper_id_list) < top_k:
                for label, papers in label_group.items():
                    if papers:
                        paper_id_list.append(papers.pop(0))
                        if len(paper_id_list) >= top_k:
                            break
            return paper_id_list

    def cosine_similarity_search(self, embedding, k=1, type_name="background_embedding"):
        """Retrieve papers through embedding
        Args:
            embedding: the input embedding
        Returns:
            result (List of Papers): return related papers with the least embedding distance
        """
        result = self.paper_client.cosine_similarity_search(
            embedding, k, type_name=type_name
        )
        # backtrack: first is itself
        result = result[1:]
        return result

    def cluster_algorithm(self, paper_id_list, similarity_matrix):
        """
        """
        threshold = self.config.RETRIEVE.similarity_threshold
        uf = UnionFind(len(paper_id_list))
        # merge
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] >= threshold:
                    if can_merge(uf, similarity_matrix, i, j, threshold):
                        uf.union(i, j)
        cluster_labels = [uf.find(i) for i in range(len(similarity_matrix))]
        return cluster_labels

    def eval_related_paper_in_all(self, score_all_dict, target_paper_id_list):
        score_all_dict = dict(
            sorted(score_all_dict.items(), key=lambda item: item[1], reverse=True)
        )
        result = {}
        related_paper_id_list = list(score_all_dict.keys())
        if len(related_paper_id_list) == 0:
            for k in self.config.RETRIEVE.top_k_list:
                result[k] = {"recall": 0, "precision": 0}
            return result, 0, 0, 0
        
        ## merge retrieved papers and target papers and clustering
        ## clustering according to the combination of background, contribution, and summary_embedding
        all_paper_id_set = set(related_paper_id_list)
        all_paper_id_set.update(target_paper_id_list)
        all_paper_id_list = list(all_paper_id_set)
        # get all target papers' background_embedding
        paper_embedding_list = [
            self.paper_client.get_paper_attribute(paper_id, f"background_embedding{self.embedding_postfix}")
            for paper_id in target_paper_id_list
        ]
        paper_embedding = np.array(paper_embedding_list)
        # get all target papers' contribution_embedding
        paper_embedding_list = [
            self.paper_client.get_paper_attribute(paper_id, f"contribution_embedding{self.embedding_postfix}")
            for paper_id in target_paper_id_list
        ]
        paper_contribution_embedding = np.array(paper_embedding_list)
        # get all target papers' summary_embedding
        paper_embedding_list = [
            self.paper_client.get_paper_attribute(paper_id, f"summary_embedding{self.embedding_postfix}")
            for paper_id in target_paper_id_list
        ]
        # abstract embedding
        paper_embedding_list = [
            self.paper_client.get_paper_attribute(paper_id, f"abstract_embedding{self.embedding_postfix}")
            for paper_id in target_paper_id_list
        ]
        paper_abstract_embedding = np.array(paper_embedding_list)

        paper_summary_embedding = np.array(paper_embedding_list)
        weight_background = self.config.RETRIEVE.s_bg
        weight_contribution = self.config.RETRIEVE.s_contribution
        weight_summary = self.config.RETRIEVE.s_summary
        weight_abstract = self.config.RETRIEVE.s_abstract
        # 2D matrix of size [# of target papers, embedding dimension]
        target_paper_embedding = (
            weight_background * paper_embedding
            + weight_contribution * paper_contribution_embedding
            + weight_summary * paper_summary_embedding
            + weight_abstract * paper_abstract_embedding
        )
        similarity_threshold = self.config.RETRIEVE.similarity_threshold
        similarity_matrix = np.dot(target_paper_embedding, target_paper_embedding.T)
        # return each target_paper's cluster label
        target_labels = self.cluster_algorithm(target_paper_id_list, similarity_matrix)
        target_paper_label_dict = dict(zip(target_paper_id_list, target_labels))
        logger.debug("Target paper cluster result: {}".format(target_paper_label_dict))
        logger.debug(
            {
                paper_id: self.paper_client.get_paper_attribute(paper_id, "title")
                for paper_id in target_paper_label_dict.keys()
            }
        )
        
        ## calculate the similarity between each two papers
        all_labels = []
        for paper_id in all_paper_id_list:
            # for each paper, get its background_embedding
            paper_bg_embedding = [
                self.paper_client.get_paper_attribute(paper_id, f"background_embedding{self.embedding_postfix}")
            ]
            paper_bg_embedding = np.array(paper_bg_embedding)
            # for each paper, get its contribution_embedding
            paper_contribution_embedding = [
                self.paper_client.get_paper_attribute(
                    paper_id, f"contribution_embedding{self.embedding_postfix}"
                )
            ]
            paper_contribution_embedding = np.array(paper_contribution_embedding)
            # for each paper, get its summary_embedding
            paper_summary_embedding = [
                self.paper_client.get_paper_attribute(paper_id, f"summary_embedding{self.embedding_postfix}")
            ]
            paper_summary_embedding = np.array(paper_summary_embedding)
            # for each paper, get its abstract_embedding
            paper_abstract_embedding = [
                self.paper_client.get_paper_attribute(paper_id, f"abstract_embedding{self.embedding_postfix}")
            ]
            paper_abstract_embedding = np.array(paper_abstract_embedding)
            
            paper_embedding = (
                weight_background * paper_bg_embedding
                + weight_contribution * paper_contribution_embedding
                + weight_summary * paper_summary_embedding
                + weight_abstract * paper_abstract_embedding
            )

            # vector of size embedding dimension
            similarities = cosine_similarity(paper_embedding, target_paper_embedding)[0]
            if np.any(similarities >= similarity_threshold):
                all_labels.append(target_labels[np.argmax(similarities)])
            else:
                all_labels.append(-1)  # other class: -1
        all_paper_label_dict = dict(zip(all_paper_id_list, all_labels))
        all_label_counts = Counter(all_paper_label_dict.values())
        logger.debug(f"All labels and the number of papers of each label: {all_label_counts}")
        target_label_counts = Counter(target_paper_label_dict.values())
        logger.debug(f"All labels and the number of target papers of each label : {target_label_counts}")
        target_label_list = list(target_label_counts.keys())
        max_k = max(self.config.RETRIEVE.top_k_list)
        logger.info("=== Begin filter related paper ===")
        max_k_paper_id_list = self.filter_related_paper(score_all_dict, top_k=max_k)
        logger.info("=== End filter related paper ===")
        ## calculate recall and precision of first {10, 20, 30, ...} papers
        for k in self.config.RETRIEVE.top_k_list:
            # 前top k 的文章
            top_k = min(k, len(max_k_paper_id_list))
            top_k_paper_id_list = max_k_paper_id_list[:top_k]
            top_k_paper_label_dict = {}
            for paper_id in top_k_paper_id_list:
                top_k_paper_label_dict[paper_id] = all_paper_label_dict[paper_id]
            logger.debug(
                "=== ideal top {}, real top {} paper id list : {}".format(k, top_k, top_k_paper_label_dict)
            )
            logger.debug(
                {
                    paper_id: self.paper_client.get_paper_attribute(paper_id, "title")
                    for paper_id in top_k_paper_label_dict.keys()
                }
            )
            top_k_label_counts = Counter(top_k_paper_label_dict.values())
            logger.debug(f"Retrieved {top_k} papers have K different label: {top_k_label_counts}")
            top_k_label_list = list(top_k_label_counts.keys())
            match_label_list = list(set(target_label_list) & set(top_k_label_list))
            logger.debug(f"match label list : {match_label_list}")
            recall = 0
            precision = 0
            for label in match_label_list:
                recall += target_label_counts[label]
            for label in match_label_list:
                precision += top_k_label_counts[label]
            recall /= len(target_paper_id_list)
            precision /= len(top_k_paper_id_list)
            result[k] = {"recall": recall, "precision": precision}

        ## calculate recall and precision of all retrieved papers
        related_paper_id_list = list(score_all_dict.keys())
        related_paper_label_dict = {}
        for paper_id in related_paper_id_list:
            related_paper_label_dict[paper_id] = all_paper_label_dict[paper_id]
        related_label_counts = Counter(related_paper_label_dict.values())
        logger.debug(f"top K label counts : {related_label_counts}")
        related_label_list = list(related_label_counts.keys())
        match_label_list = list(set(target_label_list) & set(related_label_list))
        recall = 0
        precision = 0
        for label in match_label_list:
            recall += target_label_counts[label]
        for label in match_label_list:
            precision += related_label_counts[label]
        recall /= len(target_paper_id_list)
        precision /= len(related_paper_id_list)

        logger.debug(result)
        return result, len(target_label_counts), recall, precision


class RetrieverFactory(object):
    """RetrieverFactory is a singleton class, which will return cls._instance if it has been 
    created, it saves all Retriever instances.
    Args:
        None
    Returns:
        The singleton instance of the RetrieverFactory
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RetrieverFactory, cls).__new__(
                    cls, *args, **kwargs
                )
                cls._instance.init_factory()
        return cls._instance

    def init_factory(self):
        self.retriever_classes = {}

    @staticmethod
    def get_retriever_factory():
        """The method can also return the singleton instance of the RetrieverFactory
        Args:
            None
        Returns:
            The singleton instance of the RetrieverFactory
        """
        if RetrieverFactory._instance is None:
            RetrieverFactory._instance = RetrieverFactory()
        return RetrieverFactory._instance

    def register_retriever(self, retriever_name, retriever_class) -> bool:
        """Register a new retriever class (not instance) to the RetrieverFactory
        Args:
            retriever_name: str
            retriever_class: a class object (not instance)
        Returns:
            True if add successfully, False otherwise
        """
        if retriever_name not in self.retriever_classes:
            self.retriever_classes[retriever_name] = retriever_class
            return True
        else:
            return False

    def delete_retriever(self, retriever_name) -> bool:
        if retriever_name in self.retriever_classes:
            self.retriever_classes[retriever_name] = None
            del self.retriever_classes[retriever_name]
            return True
        else:
            return False

    def __getitem__(self, key):
        return self.retriever_classes[key]

    def __len__(self):
        return len(self.retriever_classes)

    def create_retriever(self, retriever_name, *args, **kwargs) -> Retriever:
        """Return a retriever instance
        Args:
            retriever_name: str
        Returns:
            The retriever
        """
        if retriever_name not in self.retriever_classes:
            raise ValueError(f"Unknown retriever type: {retriever_name}. retriever_name should be one of {self.retriever_classes.keys()}")
        else:
            return self.retriever_classes[retriever_name](*args, **kwargs)


class autoregister:
    def __init__(self, retriever_name, *args, **kwds):
        self.retriever_name = retriever_name

    def __call__(self, cls, *args, **kwds):
        if RetrieverFactory.get_retriever_factory().register_retriever(
            self.retriever_name, cls
        ):
            cls.retriever_name = self.retriever_name
            return cls
        else:
            raise KeyError()


@autoregister("SN")
class SNRetriever(Retriever):
    def __init__(self, config):
        super().__init__(config)

    def retrieve_paper(self, bg):
        """Retrieve papers P (a set) according to embeddings' similarity between the input 
        background and the backgrounds from the database. Optionally, you can also retrieve 
        papers co-cited with P.
        Args:
            bg (str): the input background
        Returns:
            result (dict):
                "background_embedding": embedding of the input background,
                "paper" (List of int): all retrieved related_papers' ids,
                "entities" (List): An empty list (TODO: remove),
                "cocite_paper" (List of int): all papers cocited with embedding-retrieved papers
        """
        entities = []
        embedding = self.embedding_model.encode(bg, device=self.device)
        sn_paper_id_list = self.cosine_similarity_search(
            embedding=embedding,
            k=self.config.RETRIEVE.sn_retrieve_paper_num,
            type_name=f"{self.config.RETRIEVE.SN_field_name}_embedding{self.embedding_postfix}"
        )
        related_paper = set()
        related_paper.update(sn_paper_id_list)
        cocite_id_set = set()
        if self.use_cocite:
            for paper_id in related_paper:
                cocite_id_set.update(
                    self.cocite.get_cocite_ids(
                        paper_id, k=self.config.RETRIEVE.cocite_top_k
                    )
                )
            related_paper = related_paper.union(cocite_id_set)
        related_paper = list(related_paper)
        logger.debug(f"paper num before filter: {len(related_paper)}")
        result = {
            f"background_embedding{self.embedding_postfix}": embedding,
            "paper": related_paper,
            "entities": entities,
            "cocite_paper": list(cocite_id_set),
        }
        return result

    def retrieve(self, bg, entities, need_evaluate=True, target_paper_id_list=[]):
        """
        Args:
            bg (str): The user input background
        Return:
            result (dict):
                "recall": recall of paper retrieval, 0 if need_evaluate==False,
                "precision": precision of paper retrieval, 0 if need_evaluate==False,,
                "filtered_recall": recall of paper retrieval after filtering, 0 if need_evaluate==False,,
                "filtered_precision": precision of paper retrieval after filtering, 0 if need_evaluate==False,,
                "related_paper": all retrieved related_papers. !!! [ The most important item ]
                "related_paper_id_list": all retrieved related_papers' ids. !!! [ The most important item ]
                "cocite_paper_id_list": retrieve_result["cocite_paper"],
                "entities": retrieve_result["entities"], always empty
                "top_k_matrix": top_k_matrix, 0 if need_evaluate==False
                "gt_reference_num": len(target_paper_id_list)
                "retrieve_paper_num": len(related_paper_id_list),
                "label_num": TODO,
        """
        if need_evaluate:
            if target_paper_id_list is None or len(target_paper_id_list) == 0:
                logger.error(
                    "If you need evaluate retriever, please input target paper is list..."
                )
            else:
                target_paper_id_list = list(set(target_paper_id_list))
        retrieve_result = self.retrieve_paper(bg)
        related_paper_id_list = retrieve_result["paper"]
        retrieve_paper_num = len(related_paper_id_list)
        # scores between the input background and all retrieved papers
        _, _, score_all_dict = self.cal_related_score(
            retrieve_result[f"background_embedding{self.embedding_postfix}"], related_paper_id_list=related_paper_id_list,
            type_name=f"{self.config.RETRIEVE.SN_field_name}_embedding{self.embedding_postfix}"
        )
        top_k_matrix = {}
        recall = 0
        precision = 0
        filtered_recall = 0
        label_num = 0
        filtered_precision = 0
        if need_evaluate:
            top_k_matrix, label_num, recall, precision = self.eval_related_paper_in_all(
                score_all_dict, target_paper_id_list
            )
            logger.debug("Top K matrix:{}".format(top_k_matrix))
            logger.debug("before filter:")
            logger.debug(f"Recall: {recall:.3f}")
            logger.debug(f"Precision: {precision:.3f}")
        ## For idea generation, only top 10 papers will be used, which has no relations with retriveal evaluation
        related_paper = self.filter_related_paper(score_all_dict, top_k=self.config.RETRIEVE.all_retrieve_paper_num)
        related_paper = self.update_related_paper(related_paper)
        result = {
            "recall": recall,
            "precision": precision,
            "filtered_recall": filtered_recall,
            "filtered_precision": filtered_precision,
            "related_paper": related_paper,
            "related_paper_id_list": related_paper_id_list,
            "cocite_paper_id_list": retrieve_result["cocite_paper"],
            "entities": retrieve_result["entities"],
            "top_k_matrix": top_k_matrix,
            "gt_reference_num": len(target_paper_id_list),
            "retrieve_paper_num": retrieve_paper_num,
            "label_num": label_num,
        }
        return result


@autoregister("KG")
class KGRetriever(Retriever):
    def __init__(self, config):
        super().__init__(config)

    def retrieve_paper(self, entities):
        """Retrieve according to entities
        """
        new_entities = self.retrieve_entities_by_enties(entities)
        logger.debug("KG entities for retriever: {}".format(new_entities))
        related_paper = set()
        for entity in new_entities:
            paper_id_set = set(self.paper_client.find_paper_by_entity(entity))
            related_paper = related_paper.union(paper_id_set)
        cocite_id_set = set()
        if self.use_cocite:
            for paper_id in related_paper:
                cocite_id_set.update(self.cocite.get_cocite_ids(paper_id))
            related_paper = related_paper.union(cocite_id_set)
        related_paper = list(related_paper)
        logger.debug(f"paper num before filter: {len(related_paper)}")
        result = {
            "paper": related_paper,
            "entities": entities,
            "cocite_paper": list(cocite_id_set),
        }
        return result

    def retrieve(self, bg, entities, need_evaluate=False, target_paper_id_list=[]):
        """
        Args:
            context: string
        Return:
            list(dict)
        """
        if need_evaluate:
            if target_paper_id_list is None or len(target_paper_id_list) == 0:
                logger.error(
                    "If you need evaluate retriever, please input target paper is list..."
                )
            else:
                target_paper_id_list = list(set(target_paper_id_list))
                logger.debug(f"target paper id list: {target_paper_id_list}")
        retrieve_result = self.retrieve_paper(entities)
        related_paper_id_list = retrieve_result["paper"]
        retrieve_paper_num = len(related_paper_id_list)
        embedding = self.embedding_model.encode(bg, device=self.device)
        _, _, score_all_dict = self.cal_related_score(
            embedding, related_paper_id_list=related_paper_id_list,
            type_name=f"background_embedding{self.embedding_postfix}"
        )
        top_k_matrix = {}
        recall = 0
        precision = 0
        filtered_recall = 0
        label_num = 0
        filtered_precision = 0
        if need_evaluate:
            top_k_matrix, label_num, recall, precision = self.eval_related_paper_in_all(
                score_all_dict, target_paper_id_list
            )
            logger.debug("Top P ACC:{}".format(top_k_matrix))
            logger.debug("before filter:")
            logger.debug(f"Recall: {recall:.3f}")
            logger.debug(f"Precision: {precision:.3f}")
        related_paper = self.filter_related_paper(score_all_dict, top_k=self.config.RETRIEVE.all_retrieve_paper_num)
        related_paper = self.update_related_paper(related_paper)
        result = {
            "recall": recall,
            "precision": precision,
            "filtered_recall": filtered_recall,
            "filtered_precision": filtered_precision,
            "related_paper": related_paper,
            "related_paper_id_list": related_paper_id_list,
            "cocite_paper_id_list": retrieve_result["cocite_paper"],
            "entities": retrieve_result["entities"],
            "top_k_matrix": top_k_matrix,
            "gt_reference_num": len(target_paper_id_list),
            "retrieve_paper_num": retrieve_paper_num,
            "label_num": label_num,
        }
        return result


@autoregister("SNKG")
class SNKGRetriever(Retriever):
    def __init__(self, config):
        super().__init__(config)

    def retrieve_paper(self, bg, entities):
        sn_entities = []
        ## 1. Retrieve papers according to the embeddings of input background
        embedding = self.embedding_model.encode(bg, device=self.device)
        sn_paper_id_list = self.cosine_similarity_search(
            embedding, k=self.config.RETRIEVE.sn_num_for_entity,
            type_name=f"{self.config.RETRIEVE.SN_field_name}_embedding{self.embedding_postfix}"
        )
        related_paper = set()
        related_paper.update(sn_paper_id_list)
        logger.debug(f"SN retrieve {len(related_paper)} papers")

        ## 2. Retrieve papers according to entites
        # Fetch all entities from embedding-retrieved papers
        sn_entities += self.paper_client.find_entities_by_paper_list(sn_paper_id_list)
        logger.debug("SN entities for retriever: {}".format(sn_entities))
        entities = list(set(entities + sn_entities))
        # Expand entity list through synonyms
        new_entities = self.retrieve_entities_by_enties(entities)
        logger.debug("SNKG entities for retriever: {}".format(new_entities))
        paper_id_set = set()
        for entity in new_entities:
            paper_id_set.update(self.paper_client.find_paper_by_entity(entity))
        related_paper = related_paper.union(paper_id_set)
        logger.debug(f"Entity retrieve {len(paper_id_set)} papers")
        logger.debug(f"SN+entity retrieve {len(related_paper)} papers")

        ## 3. Retrieve papers according to citation co-occurrence
        cocite_id_set = set()
        if self.use_cocite:
            for paper_id in related_paper:
                cocite_id_set.update(self.cocite.get_cocite_ids(paper_id))
            related_paper = related_paper.union(cocite_id_set)
        logger.debug(f"Cocite retrieve {len(cocite_id_set)} papers")
        logger.debug(f"SN+entity+cocite retrieve {len(related_paper)} papers")

        ## 4. Return retrieval results
        related_paper = list(related_paper)
        result = {
            f"background_embedding{self.embedding_postfix}": embedding,
            "paper": related_paper,
            "entities": entities,
            "cocite_paper": list(cocite_id_set),
        }
        return result

    def retrieve(
        self, bg, entities, need_evaluate=True, target_paper_id_list=[]
    ):
        """
        Args:
            context: string
        Return:
            list(dict)
        """
        if need_evaluate:
            if target_paper_id_list is None or len(target_paper_id_list) == 0:
                logger.error(
                    "If you need evaluate retriever, please input target paper is list..."
                )
            else:
                target_paper_id_list = list(set(target_paper_id_list))
                logger.debug(f"target paper id list: {target_paper_id_list}")
        retrieve_result = self.retrieve_paper(bg, entities)
        related_paper_id_list = retrieve_result["paper"]
        retrieve_paper_num = len(related_paper_id_list)
        logger.info("=== Begin cal related paper score ===")
        _, _, score_all_dict = self.cal_related_score(
            retrieve_result[f"background_embedding{self.embedding_postfix}"], related_paper_id_list=related_paper_id_list,
            type_name=f"background_embedding{self.embedding_postfix}"
        )
        logger.info("=== End cal related paper score ===")
        top_k_matrix = {}
        recall = 0
        precision = 0
        filtered_recall = 0
        filtered_precision = 0
        label_num = 0
        if need_evaluate:
            top_k_matrix, label_num, recall, precision = self.eval_related_paper_in_all(
                score_all_dict, target_paper_id_list
            )
            logger.debug("Top K matrix:{}".format(top_k_matrix))
            logger.debug("before filter:")
            logger.debug(f"Recall: {recall:.3f}")
            logger.debug(f"Precision: {precision:.3f}")
        logger.info("=== Begin filter related paper score ===")
        related_paper = self.filter_related_paper(score_all_dict, self.config.RETRIEVE.all_retrieve_paper_num)
        logger.info("=== End filter related paper score ===")
        related_paper = self.update_related_paper(related_paper)
        result = {
            "recall": recall,
            "precision": precision,
            "filtered_recall": filtered_recall,
            "filtered_precision": filtered_precision,
            "related_paper": related_paper,
            "cocite_paper_id_list": retrieve_result["cocite_paper"],
            "related_paper_id_list": related_paper_id_list,
            "entities": retrieve_result["entities"],
            "top_k_matrix": top_k_matrix,
            "gt_reference_num": (
                len(target_paper_id_list) if target_paper_id_list is not None else 0
            ),
            "retrieve_paper_num": retrieve_paper_num,
            "label_num": label_num,
        }
        return result
