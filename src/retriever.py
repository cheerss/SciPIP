import os
from utils.paper_retriever import RetrieverFactory
from utils.llms_api import APIHelper
from utils.paper_client import PaperClient
from utils.header import ConfigReader
from omegaconf import OmegaConf
import click
import json
from loguru import logger
import warnings
from utils.hash import check_env, check_embedding

warnings.filterwarnings("ignore")


@click.group()
@click.pass_context
def main(ctx):
    """
    Evaluate Retriever SN/KG/SNKG
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
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
    default="assets/data/test_acl_2024.json",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.pass_context
def retrieve(ctx,
    config_path, ids_path
): 
    initial_kwargs={ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}
    kwargs = {"RETRIEVE": {}, "DEFAULT": {}}
    for k, v in initial_kwargs.items():
        if "num" in k:
            kwargs["RETRIEVE"][k] = int(v)
        elif "s_" in k:
            kwargs["RETRIEVE"][k] = float(v)
        elif "use_cocite" in k:
            kwargs["RETRIEVE"][k] = bool(int(v))
        else:
            kwargs["RETRIEVE"][k] = v
    config = ConfigReader.load(config_path, **kwargs)
    check_embedding(config.DEFAULT.embedding)
    check_env()
    log_dir = config.DEFAULT.log_dir
    retriever_name = config.RETRIEVE.retriever_name
    cluster_to_filter = config.RETRIEVE.use_cluster_to_filter
    co_cite = config.RETRIEVE.use_cocite
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")
    log_file = os.path.join(
        log_dir,
        "retriever_eval_{}_cocite-{}_cluster-{}.log".format(
            retriever_name, co_cite, cluster_to_filter
        ),
    )
    logger.add(log_file, level=config.DEFAULT.log_level)
    logger.info("=== Retriever name : {} ===".format(retriever_name))
    logger.info("Loaded configuration:\n{}".format(OmegaConf.to_yaml(config)))
    api_helper = APIHelper(config)
    paper_client = PaperClient()
    precision = 0
    filtered_precision = 0
    recall = 0
    filtered_recall = 0
    num = 0
    gt_reference_num = 0
    retrieve_paper_num = 0
    label_num = 0
    top_k_precision = {p: 0 for p in config.RETRIEVE.top_k_list}
    top_k_recall = {p: 0 for p in config.RETRIEVE.top_k_list}
    # Init Retriever
    rt = RetrieverFactory.get_retriever_factory().create_retriever(
        retriever_name,
        config
    )
    for line in ids_path:
        paper = json.loads(line)
        logger.info("\nbegin generate paper hash id {}".format(paper["hash_id"]))
        # 1. Get Background
        paper = paper_client.get_paper_by_id(paper["hash_id"])
        if "background" in paper.keys():
            bg = paper["background"]
        else:
            logger.error(f"paper hash_id {paper['hash_id']} doesn't have background...")
            continue
        if "entities" in paper.keys():
            entities = paper["entities"]
        else:
            entities = api_helper.generate_entity_list(bg)
        logger.info("\norigin entities from background: {}".format(entities))
        cite_type = config.RETRIEVE.cite_type
        if cite_type in paper and len(paper[cite_type]) >= 5:
            target_paper_id_list = paper[cite_type]
        else:
            logger.warning(
                "hash_id {} cite paper num less than 5 ...".format(paper["hash_id"])
            )
            continue
        # 2. Retrieve
        result = rt.retrieve(
            bg, entities, need_evaluate=True, target_paper_id_list=target_paper_id_list
        )
        filtered_precision += result["filtered_precision"]
        precision += result["precision"]
        filtered_recall += result["filtered_recall"]
        gt_reference_num += result["gt_reference_num"]
        retrieve_paper_num += result["retrieve_paper_num"]
        recall += result["recall"]
        label_num += result["label_num"]
        for k, v in result["top_k_matrix"].items():
            top_k_recall[k] += v["recall"]
            top_k_precision[k] += v["precision"]
        num += 1
        if num >= 100:
            break
        continue
    logger.info("=== Finish Report ===")
    logger.info(f"{'Test Paper Num:':<25} {num}")
    logger.info(f"{'Average Precision:':<25} {precision/num:.3f}")
    logger.info(f"{'Average Recall:':<25} {recall/num:.3f}")
    logger.info(f"{'Average GT Ref Paper Num:':<25} {gt_reference_num/num:.3f}")
    logger.info(f"{'Average Retrieve Paper Num:':<25} {retrieve_paper_num/num:.3f}")
    logger.info(f"{'Average Label Num:':<25} {label_num/num:.3f}")
    # Print Eval Result
    logger.info("=== Top-K Metrics ===")
    logger.info(
        f"=== USE_COCIT: {co_cite}, USE_CLUSTER_TO_FILTER: {cluster_to_filter} ==="
    )
    logger.info("| Top K  | Recall | Precision |")
    logger.info("|--------|--------|-----------|")
    for k in config.RETRIEVE.top_k_list:
        if k <= retrieve_paper_num / num:
            logger.info(
                f"| {k:<5} | {top_k_recall[k]/num:.3f}  | {top_k_precision[k]/num:.3f}    |"
            )
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
