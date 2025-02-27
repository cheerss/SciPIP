import re
import os
import hashlib
import torch
import struct
from collections import Counter
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from .header import get_dir

ENV_CHECKED = False
EMBEDDING_CHECKED = False


def check_embedding(repo_id):
    print("=== check embedding model ===")
    global EMBEDDING_CHECKED
    if not EMBEDDING_CHECKED:
        # Define the repository and files to download
        local_dir = f"./assets/model/{repo_id}"
        if repo_id in [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/llm-embedder",
        ]:
            # repo_id = "sentence-transformers/all-MiniLM-L6-v2"
            # repo_id = "BAAI/bge-small-en-v1.5"
            files_to_download = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer_config.json",
                "vocab.txt",
            ]
        elif repo_id in [
            "jina-embeddings-v3",
        ]:
            files_to_download = [
                "model.safetensors",
                "modules.json",
                "tokenizer.json",
                "config_sentence_transformers.json",
                "tokenizer_config.json",
                "1_Pooling/config.json",
                "config.json",
            ]
        elif repo_id in ["Alibaba-NLP/gte-base-en-v1.5"]:
            files_to_download = [
                "config.json",
                "model.safetensors",
                "modules.json",
                "tokenizer.json",
                "sentence_bert_config.json",
                "tokenizer_config.json",
                "vocab.txt",
            ]
        # Download each file and save it to the /model/bge directory
        for file_name in files_to_download:
            if not os.path.exists(os.path.join(local_dir, file_name)):
                print(
                    f"file: {file_name} not exist in {local_dir}, try to download from huggingface ..."
                )
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    local_dir=local_dir,
                )
        EMBEDDING_CHECKED = True


def check_env():
    global ENV_CHECKED
    if not ENV_CHECKED:
        env_name_list = [
            "NEO4J_URL",
            "NEO4J_USERNAME",
            "NEO4J_PASSWD",
            "MODEL_NAME",
            "MODEL_TYPE",
            "BASE_URL",
        ]
        for env_name in env_name_list:
            if env_name not in os.environ or os.environ[env_name] == "":
                raise ValueError(f"{env_name} is not set...")
        if os.environ["MODEL_TYPE"] != "Local":
            env_name = "MODEL_API_KEY"
            if env_name not in os.environ or os.environ[env_name] == "":
                raise ValueError(f"{env_name} is not set...")
        ENV_CHECKED = True


class EmbeddingModel:
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            local_dir = f"./assets/model/{config.DEFAULT.embedding}"
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.embedding_model = SentenceTransformer(
                model_name_or_path=get_dir(local_dir),
                device=device,
                trust_remote_code=True,
            )
            if "jina-embeddings-v3" in config.DEFAULT.embedding:
                cls._instance.embedding_model[0].default_task = config.DEFAULT.embedding_task
            print(f"==== using device {device} ====")
        return cls._instance


def get_embedding_model(config):
    print("=== get embedding model ===")
    check_embedding(config.DEFAULT.embedding)
    return EmbeddingModel(config).embedding_model


def generate_hash_id(input_string):
    if input_string is None:
        return None
    sha1_hash = hashlib.sha256(input_string.lower().encode("utf-8")).hexdigest()
    binary_hash = bytes.fromhex(sha1_hash)
    int64_hash = struct.unpack(">q", binary_hash[:8])[0]
    return abs(int64_hash)


def extract_ref_id(text, references):
    """
    references: paper["references"]
    """
    # 正则表达式模式，用于匹配[数字, 数字]格式
    pattern = r"\[\d+(?:,\s*\d+)*\]"
    # 提取所有匹配的内容
    ref_list = re.findall(pattern, text)
    # ref ['[15, 16]', '[5]', '[2, 3, 8]']
    combined_ref_list = []
    if len(ref_list) > 0:
        # 说明是pattern 0
        for ref in ref_list:
            # 移除方括号并分割数字
            numbers = re.findall(r"\d+", ref)
            # 将字符串数字转换为整数并加入到列表中
            combined_ref_list.extend(map(int, numbers))
        # 去重并排序
        ref_counts = Counter(combined_ref_list)
        ref_counts = dict(sorted(ref_counts.items()))
        # 对多个，只保留引用最多的一个
        for ref in ref_list:
            # 移除方括号并分割数字
            numbers = re.findall(r"\d+", ref)
            # 找到只引用了一次的
            temp_list = []
            for num in numbers:
                num = int(num)
                if ref_counts[num] == 1:
                    temp_list.append(num)
            if len(temp_list) == len(numbers):
                temp_list = temp_list[1:]
            for num in temp_list:
                del ref_counts[num]
    hash_id_list = []
    for idx in ref_counts.keys():
        hash_id_list.append(generate_hash_id(references[idx]))
    return hash_id_list


if __name__ == "__main__":
    # 示例用法
    input_string = "example_string"
    hash_id = generate_hash_id(input_string)
    print("INT64 Hash ID:", hash_id)
