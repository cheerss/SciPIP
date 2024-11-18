import os
import re
import json
import torch
from tqdm import tqdm
from neo4j import GraphDatabase
from collections import defaultdict, deque
from py2neo import Graph, Node, Relationship
from loguru import logger

class PaperClient:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PaperClient, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self.driver = self.get_neo4j_driver()
            self.teb_model = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            PaperClient._initialized = True

    def get_neo4j_driver(self):
        URI = os.environ["NEO4J_URL"]
        NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
        NEO4J_PASSWD = os.environ["NEO4J_PASSWD"]
        AUTH = (NEO4J_USERNAME, NEO4J_PASSWD)
        driver = GraphDatabase.driver(URI, auth=AUTH)
        return driver

    def update_paper_from_client(self, paper):
        paper_id = paper["hash_id"]
        if paper_id is None:
            return None
        query = f"""
            MATCH (p:Paper {{hash_id: {paper_id}}})
            RETURN p
            """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query).data())
        if result:
            paper_from_client = result[0]['p']
            if paper_from_client is not None:
                paper.update(paper_from_client)
    
    def get_paper_attribute(self, paper_id, attribute_name):
        query = f"""
            MATCH (p:Paper {{hash_id: {paper_id}}})
            RETURN p.{attribute_name} AS attributeValue
            """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query).data())
        if result:
            return result[0]['attributeValue']
        else:
            logger.error(f"paper id {paper_id} get {attribute_name} failed.")
            return None
    
    def get_paper_by_attribute(self, attribute_name, anttribute_value):
        query = f"""
            MATCH (p:Paper {{{attribute_name}: '{anttribute_value}'}})
            RETURN p
            """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query).data())
        if result:
            return result[0]['p']
        else:
            return None

    def get_paper_from_term(self, entity):
        if entity is None:
            return None
        query = """
            MATCH (p:Paper)
            WHERE p.entity = $entity
            RETURN p.hash_id as hash_id
            """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, entity=entity).data())
        if result:
            return [record['hash_id'] for record in result]
        else:
            return []
    
    def find_related_entities_by_entity(self, entity_name, n=1, k=3, relation_name="related"):
        # relation_name = "related"
        def bfs_query(entity_name, n, k):
            queue = deque([(entity_name, 0)])  
            visited = set([entity_name])  
            related_entities = set()  

            while queue:
                batch_queue = [queue.popleft() for _ in range(len(queue))]  
                batch_entities = [item[0] for item in batch_queue]
                batch_depths = [item[1] for item in batch_queue]

                if all(depth >= n for depth in batch_depths):
                    continue
                if relation_name == "related":
                    query = """
                        UNWIND $batch_entities AS entity_name
                        MATCH (e1:Entity {name: entity_name})-[:RELATED_TO]->(p:Paper)<-[:RELATED_TO]-(e2:Entity)
                        WHERE e1 <> e2
                        WITH e1, e2, COUNT(p) AS common_papers, entity_name
                        WHERE common_papers > $k
                        RETURN e2.name AS entities, entity_name AS source_entity, common_papers
                    """
                elif relation_name == "connect":
                    query = """
                        UNWIND $batch_entities AS entity_name
                        MATCH (e1:Entity {name: entity_name})-[r:CONNECT]-(e2:Entity)
                        WHERE e1 <> e2 and r.strength >= $k
                        WITH e1, e2, entity_name
                        RETURN e2.name AS entities, entity_name AS source_entity
                    """
                with self.driver.session() as session:
                    result = session.execute_read(lambda tx: tx.run(query, batch_entities=batch_entities, k=k).data())

                for record in result:
                    entity = record['entities']
                    source_entity = record['source_entity']
                    if entity not in visited:
                        visited.add(entity)
                        queue.append((entity, batch_depths[batch_entities.index(source_entity)] + 1))
                        related_entities.add(entity)

            return list(related_entities)
        related_entities = bfs_query(entity_name, n, k)
        if entity_name in related_entities:
            related_entities.remove(entity_name)
        return related_entities

    def find_entities_by_paper(self, hash_id: int):
        query = """
            MATCH (e:Entity)-[:RELATED_TO]->(p:Paper {hash_id: $hash_id})
            RETURN e.name AS entity_name
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, hash_id=hash_id).data())
        if result:
            return [record['entity_name'] for record in result]
        else:
            return []

    def find_paper_by_entity(self, entity_name):
        query = """
            MATCH (e1:Entity {name: $entity_name})-[:RELATED_TO]->(p:Paper)
            RETURN p.hash_id AS hash_id
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, entity_name=entity_name).data())
        if result:
            return [record['hash_id'] for record in result]
        else:
            return []
        
    # TODO: @云翔
    # 增加通过entity返回包含entity语句的功能
    def find_sentence_by_entity(self, entity_name):
        # Return: list(str)
        return []
    

    def find_sentences_by_entity(self, entity_name):
        query = """
        MATCH (e:Entity {name: $entity_name})-[:RELATED_TO]->(p:Paper)
        WHERE p.abstract CONTAINS $entity_name OR
          p.introduction CONTAINS $entity_name OR
          p.methodology CONTAINS $entity_name
        RETURN p.abstract AS abstract, 
            p.introduction AS introduction, 
            p.methodology AS methodology, 
            p.hash_id AS hash_id
        """
        sentences = []
        
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, entity_name=entity_name).data())
        for record in result:
            for key in ['abstract', 'introduction', 'methodology']:
                if record[key]:
                    filtered_sentences = [sentence.strip() + '.' for sentence in record[key].split('.') if entity_name in sentence]
                    sentences.extend([f"{record['hash_id']}: {sentence}" for sentence in filtered_sentences])

        return sentences

    def select_paper(self, venue_name, year):
        query = """
            MATCH (n:Paper) where n.year=$year and n.venue_name=$venue_name return n
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, year=year, venue_name=venue_name).data())
        if result:
            return [record['n'] for record in result]
        else:
            return []

    def add_paper_node(self, paper: dict):
        if "summary" not in paper.keys():
            paper["summary"] = None
        if "abstract" not in paper.keys():
            paper["abstract"] = None
        if "introduction" not in paper.keys():
            paper["introduction"] = None
        if "reference" not in paper.keys():
            paper["reference"] = None
        if "cite" not in paper.keys():
            paper["cite"] = None
        if "motivation" not in paper.keys():
            paper["motivation"] = None
        if "contribution" not in paper.keys():
            paper["contribution"] = None
        if "methodology" not in paper.keys():
            paper["methodology"] = None
        if "ground_truth" not in paper.keys():
            paper["ground_truth"] = None
        if "reference_filter" not in paper.keys():
            paper["reference_filter"] = None
        if "conclusions" not in paper.keys():
            paper["conclusions"] = None
        query = """
            MERGE (p:Paper {hash_id: $hash_id})
            ON CREATE SET p.venue_name = $venue_name, p.year = $year, p.title = $title, p.pdf_url = $pdf_url, p.abstract = $abstract, p.introduction = $introduction, p.reference = $reference, p.summary = $summary, p.motivation = $motivation, p.contribution = $contribution, p.methodology = $methodology, p.ground_truth = $ground_truth, p.reference_filter = $reference_filter, p.conclusions = $conclusions
            ON MATCH SET p.venue_name = $venue_name, p.year = $year, p.title = $title, p.pdf_url = $pdf_url, p.abstract = $abstract, p.introduction = $introduction, p.reference = $reference, p.summary = $summary, p.motivation = $motivation, p.contribution = $contribution, p.methodology = $methodology, p.ground_truth = $ground_truth, p.reference_filter = $reference_filter, p.conclusions = $conclusions
            RETURN p
            """
        with self.driver.session() as session:
            result = session.execute_write(lambda tx: tx.run(query, hash_id=paper["hash_id"], venue_name=paper["venue_name"], year=paper["year"], title=paper["title"], pdf_url=paper["pdf_url"], abstract=paper["abstract"], introduction=paper["introduction"], reference=paper["reference"], summary=paper["summary"], motivation=paper["motivation"], contribution=paper["contribution"], methodology=paper["methodology"], ground_truth=paper["ground_truth"], reference_filter=paper["reference_filter"], conclusions=paper["conclusions"]).data())

    def check_entity_node_count(self, hash_id: int):
        query_check_count = """
            MATCH (e:Entity)-[:RELATED_TO]->(p:Paper {hash_id: $hash_id})
            RETURN count(e) AS entity_count
        """
        with self.driver.session() as session:
            # Check the number of related entities
            result = session.execute_read(lambda tx: tx.run(query_check_count, hash_id=hash_id).data())
            if result[0]["entity_count"] > 3:
                return False
        return True

    def add_entity_node(self, hash_id: int, entities: list):
        query = """
            MERGE (e:Entity {name: $entity_name})
            WITH e
            MATCH (p:Paper {hash_id: $hash_id})
            MERGE (e)-[:RELATED_TO]->(p)
            RETURN e, p
        """
        with self.driver.session() as session:
            for entity_name in entities:
                result = session.execute_write(lambda tx: tx.run(query, entity_name=entity_name, hash_id=hash_id).data())
        
    def add_paper_citation(self, paper: dict):
        query = """
            MERGE (p:Paper {hash_id: $hash_id}) ON MATCH SET p.cite_id_list = $cite_id_list, p.entities = $entities, p.all_cite_id_list = $all_cite_id_list
            """
        with self.driver.session() as session:
            result = session.execute_write(lambda tx: tx.run(query, hash_id=paper["hash_id"], cite_id_list=paper["cite_id_list"], entities=paper["entities"], all_cite_id_list=paper["all_cite_id_list"]).data())

    def add_paper_abstract_embedding(self, embedding_model, hash_id=None):
        if hash_id is not None:
            query = """
            MATCH (p:Paper {hash_id: $hash_id})
            WHERE p.abstract IS NOT NULL
            RETURN p.abstract AS context, p.hash_id AS hash_id, p.title AS title
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id).data())
        else:
            query = """
            MATCH (p:Paper)
            WHERE p.abstract IS NOT NULL
            RETURN p.abstract AS context, p.hash_id AS hash_id, p.title AS title
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query).data())
        contexts = [result["title"] + result["context"] for result in results]
        paper_ids = [result["hash_id"] for result in results]
        context_embeddings = embedding_model.encode(contexts, batch_size=512, convert_to_tensor=True, device=self.device)
        query = """
            MERGE (p:Paper {hash_id: $hash_id})
            ON CREATE SET p.abstract_embedding = $embedding
            ON MATCH SET p.abstract_embedding = $embedding
        """
        for idx, hash_id in tqdm(enumerate(paper_ids)):
            embedding = context_embeddings[idx].detach().cpu().numpy().flatten().tolist()
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id, embedding=embedding).data())

    def add_paper_bg_embedding(self, embedding_model, hash_id=None):
        if hash_id is not None:
            query = """
            MATCH (p:Paper {hash_id: $hash_id})
            WHERE p.motivation IS NOT NULL
            RETURN p.motivation AS context, p.hash_id AS hash_id
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id).data())
        else:
            query = """
            MATCH (p:Paper)
            WHERE p.motivation IS NOT NULL
            RETURN p.motivation AS context, p.hash_id AS hash_id
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query).data())
        contexts = [result["context"] for result in results]
        paper_ids = [result["hash_id"] for result in results]
        context_embeddings = embedding_model.encode(contexts, batch_size=256, convert_to_tensor=True, device=self.device)
        query = """
            MERGE (p:Paper {hash_id: $hash_id})
            ON CREATE SET p.embedding = $embedding
            ON MATCH SET p.embedding = $embedding
        """
        for idx, hash_id in tqdm(enumerate(paper_ids)):
            embedding = context_embeddings[idx].detach().cpu().numpy().flatten().tolist()
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id, embedding=embedding).data())

    def add_paper_contribution_embedding(self, embedding_model, hash_id=None):
        if hash_id is not None:
            query = """
            MATCH (p:Paper {hash_id: $hash_id})
            WHERE p.contribution IS NOT NULL
            RETURN p.contribution AS context, p.hash_id AS hash_id
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id).data())
        else:
            query = """
            MATCH (p:Paper)
            WHERE p.contribution IS NOT NULL
            RETURN p.contribution AS context, p.hash_id AS hash_id
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query).data())
        contexts = [result["context"] for result in results]
        paper_ids = [result["hash_id"] for result in results]
        context_embeddings = embedding_model.encode(contexts, batch_size=256, convert_to_tensor=True, device=self.device)
        query = """
            MERGE (p:Paper {hash_id: $hash_id})
            ON CREATE SET p.contribution_embedding = $embedding
            ON MATCH SET p.contribution_embedding = $embedding
        """
        for idx, hash_id in tqdm(enumerate(paper_ids)):
            embedding = context_embeddings[idx].detach().cpu().numpy().flatten().tolist()
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id, embedding=embedding).data())
    

    def add_paper_summary_embedding(self, embedding_model, hash_id=None):
        if hash_id is not None:
            query = """
            MATCH (p:Paper {hash_id: $hash_id})
            WHERE p.summary IS NOT NULL
            RETURN p.summary AS context, p.hash_id AS hash_id
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id).data())
        else:
            query = """
            MATCH (p:Paper)
            WHERE p.summary IS NOT NULL
            RETURN p.summary AS context, p.hash_id AS hash_id
            """
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query).data())
        contexts = [result["context"] for result in results]
        paper_ids = [result["hash_id"] for result in results]
        context_embeddings = embedding_model.encode(contexts, batch_size=256, convert_to_tensor=True, device=self.device)
        query = """
            MERGE (p:Paper {hash_id: $hash_id})
            ON CREATE SET p.summary_embedding = $embedding
            ON MATCH SET p.summary_embedding = $embedding
        """
        for idx, hash_id in tqdm(enumerate(paper_ids)):
            embedding = context_embeddings[idx].detach().cpu().numpy().flatten().tolist()
            with self.driver.session() as session:
                results = session.execute_write(lambda tx: tx.run(query, hash_id=hash_id, embedding=embedding).data())
        
    def cosine_similarity_search(self, embedding, k=1, type_name="embedding"):
        query = f"""
            MATCH (paper:Paper)
            WITH paper,
                vector.similarity.cosine(paper.{type_name}, $embedding) AS score
            WHERE score > 0
            RETURN paper, score
            ORDER BY score DESC LIMIT {k}
            """
        with self.driver.session() as session:
            results = session.execute_read(lambda tx: tx.run(query, embedding=embedding).data())
        related_paper = [] 
        for result in results:
            related_paper.append(result["paper"]["hash_id"])
        return related_paper

    def create_vector_index(self):
        """
        适用于Paper节点
        针对Paper节点上的是属性 embedding 进行索引
        索引向量的维度为384
        适用余弦相似度作为计算相似度的方法
        """
        query = """
            CREATE VECTOR INDEX `paper-embeddings`
            FOR (n:Paper) ON (n.embedding)
            OPTIONS {indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
            }}
            """
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query).data())
    
    def filter_paper_id_list(self, paper_id_list, year="2024"):
        if not paper_id_list:
            return []
        # WHERE p.year < "2024" AND p.venue_name <> "acl"
        query = """
            UNWIND $paper_id_list AS hash_id
            MATCH (p:Paper {hash_id: hash_id})
            WHERE p.year < $year
            RETURN p.hash_id AS hash_id
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, paper_id_list=paper_id_list, year=year).data())
        
        existing_paper_ids = [record['hash_id'] for record in result]
        existing_paper_ids = list(set(existing_paper_ids))
        return existing_paper_ids
    
    def check_index_exists(self):
        query = "SHOW INDEXES"
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query).data())
        for record in result:
            if record["name"] == "paper-embeddings":
                return True
        return False

    def clear_database(self):
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query).data())
    
    def get_entity_related_paper_num(self, entity_name):
        query = """
            MATCH (e:Entity {name: $entity_name})-[:RELATED_TO]->(p:Paper)
            WITH COUNT(p) AS PaperCount
            RETURN PaperCount
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, entity_name=entity_name).data())
        paper_num = result[0]['PaperCount']
        return paper_num

    def get_entity_text(self):
        query = """
            MATCH (e:Entity)-[:RELATED_TO]->(p:Paper)
            WHERE p.venue_name = $venue_name and p.year = $year
            WITH p, collect(e.name) AS entity_names
            RETURN p, reduce(text = '', name IN entity_names | text + ' ' + name) AS entity_text
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query).data())
        text_list = [record['entity_text'] for record in result]
        return text_list
    
    def get_entity_combinations(self, venue_name, year):
        def process_paper_relationships(session, entity_name_1, entity_name_2, abstract):
            if entity_name_2 < entity_name_1:
                entity_name_1, entity_name_2 = entity_name_2, entity_name_1
            query = """
                MATCH (e1:Entity {name: $entity_name_1})
                MATCH (e2:Entity {name: $entity_name_2})
                MERGE (e1)-[r:CONNECT]->(e2)
                ON CREATE SET r.strength = 1
                ON MATCH SET r.strength = r.strength + 1
            """
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)
            for sentence in sentences:
                sentence = sentence.lower()
                if entity_name_1 in sentence and entity_name_2 in sentence:
                    # 如果两个实体在同一句话中出现过，则创建或更新 CONNECT 关系
                    session.execute_write(
                        lambda tx: tx.run(query, entity_name_1=entity_name_1, entity_name_2=entity_name_2).data()
                    )
                    # logger.debug(f"CONNECT relation created or updated between {entity_name_1} and {entity_name_2} for Paper ID {paper_id}")
                    break  # 如果找到一次出现就可以退出循环

        query = """
            MATCH (e:Entity)-[:RELATED_TO]->(p:Paper)
            WHERE p.venue_name=$venue_name and p.year=$year
            WITH p, collect(e) as entities
            UNWIND range(0, size(entities)-2) as i
            UNWIND range(i+1, size(entities)-1) as j
            RETURN p.hash_id AS hash_id, entities[i].name AS entity_name_1, entities[j].name AS entity_name_2
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, venue_name=venue_name, year=year).data())
            for record in tqdm(result):
                paper_id = record["hash_id"]
                entity_name_1 = record['entity_name_1']
                entity_name_2 = record['entity_name_2']
                abstract = self.get_paper_attribute(paper_id, "abstract")
                process_paper_relationships(session, entity_name_1, entity_name_2, abstract)

    def build_citemap(self):
        citemap = defaultdict(set)
        query = """
            MATCH (p:Paper)
            RETURN p.hash_id AS hash_id, p.cite_id_list AS cite_id_list
        """
        with self.driver.session() as session:
            results = session.execute_read(lambda tx: tx.run(query).data())
        for result in results:
            hash_id = result['hash_id']
            cite_id_list = result['cite_id_list']
            if cite_id_list:
                for cited_id in cite_id_list:
                    citemap[hash_id].add(cited_id)
        return citemap

    def neo4j_backup(self):
        URI = os.environ["NEO4J_URL"]
        NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
        NEO4J_PASSWD = os.environ["NEO4J_PASSWD"]
        AUTH = (NEO4J_USERNAME, NEO4J_PASSWD)
        graph = Graph(URI, auth=AUTH)
        # 创建一个字典来保存数据
        data = {"nodes": [], "relationships": []}
        query = """
            MATCH (e:Entity)-[r:RELATED_TO]->(p:Paper)
            WHERE p.venue_name='iclr' and p.year='2024'
            RETURN p, e, r
        """
        results = graph.run(query)
        # 处理查询结果
        for record in tqdm(results):
            paper_node = record["p"]
            entity_node = record["e"]
            relationship = record["r"]
            # 将节点数据加入字典
            data["nodes"].append({
                "id": paper_node.identity,
                "label": "Paper",
                "properties": dict(paper_node)
            })
            data["nodes"].append({
                "id": entity_node.identity,
                "label": "Entity",
                "properties": dict(entity_node)
            })
            # 将关系数据加入字典
            data["relationships"].append({
                "start_node": entity_node.identity,
                "end_node": paper_node.identity,
                "type": "RELATED_TO",
                "properties": dict(relationship)
            })
        query = """
            MATCH (p:Paper)
            WHERE p.venue_name='acl' and p.year='2024'
            RETURN p
        """
        """
        results = graph.run(query)
        for record in tqdm(results):
            paper_node = record["p"]
            # 将节点数据加入字典
            data["nodes"].append({
                "id": paper_node.identity,
                "label": "Paper",
                "properties": dict(paper_node)
            })
        """
        # 去除重复节点
        # data["nodes"] = [dict(t) for t in {tuple(d.items()) for d in data["nodes"]}]
        unique_nodes = []
        seen = set()
        for node in tqdm(data["nodes"]):
            # 将字典项转换为不可变的元组，以便用于集合去重
            node_tuple = str(tuple(sorted(node.items())))
            if node_tuple not in seen:
                seen.add(node_tuple)
                unique_nodes.append(node)
        data["nodes"] = unique_nodes
        # 将数据保存为 JSON 文件
        with open("./assets/data/scipip_neo4j_clean_backup.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def neo4j_import_data(self):
        # clear_database() # 清空数据库，谨慎执行
        URI = os.environ["NEO4J_URL"]
        NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
        NEO4J_PASSWD = os.environ["NEO4J_PASSWD"]
        AUTH = (NEO4J_USERNAME, NEO4J_PASSWD)
        graph = Graph(URI, auth=AUTH)
        # 从 JSON 文件中读取数据
        with open("./assets/data/scipip_neo4j_clean_backup.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # 创建节点
        nodes = {}
        for node_data in data["nodes"]:
            label = node_data["label"]
            properties = node_data["properties"]
            node = Node(label, **properties)
            graph.create(node)
            nodes[node_data["id"]] = node

        # 创建关系
        for relationship_data in data["relationships"]:
            start_node = nodes[relationship_data["start_node"]]
            end_node = nodes[relationship_data["end_node"]]
            properties = relationship_data["properties"]
            rel_type = relationship_data["type"]
            relationship = Relationship(start_node, rel_type, end_node, **properties)
            graph.create(relationship)

    def get_paper_by_id(self, hash_id):
        paper = {"hash_id": hash_id}
        self.update_paper_from_client(paper)
        return paper


if __name__ == "__main__":
    paper_client = PaperClient()
    # paper_client.neo4j_backup()
    paper_client.neo4j_import_data()
