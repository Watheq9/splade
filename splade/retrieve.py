import hydra
from omegaconf import DictConfig
import os

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .evaluate import evaluate
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseRetrieval
from .utils.utils import get_dataset_name, get_initialize_config




def convert_ranking_to_trec(retr_res, doc_idx_to_key, run_path, tag='splade'):

    output = ""
    for qid in retr_res.keys():
        doc_list = retr_res[qid]
        # Sort the dictionary by its values in descending order
        doc_list = sorted(doc_list.items(), key=lambda item: item[1], reverse=True)
        rank = 1
        for (doc_idx, score) in doc_list:
            doc_idx = int(doc_idx)
            docno = doc_idx_to_key[doc_idx]
            # qid = query_idx_to_key[query_idx]
            line = f"{qid}\tQ0\t{docno}\t{rank}\t{score}\t{tag}"
            output += line + "\n"
            rank += 1

    if not os.path.exists(os.path.dirname(run_path)):
        os.makedirs(os.path.dirname(run_path), exist_ok=True)

    with open(run_path, 'w') as file: 
        file.write(output)
    return output


# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def retrieve(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)
    id_style = config.get("id_style", "row_id")
    splade_model = config.get("splade_model")


    # print(exp_dict)
    #if HF: need to udate config.
    if "hf_training" in config:
       init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
       init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None

    batch_size = 1
    # NOTE: batch_size is set to 1, currently no batched implementation for retrieval (TODO)

    model = get_model(config, init_dict)

    corpus_file = exp_dict["data"]["COLLECTION_PATH"]
    query_file = exp_dict["data"]["Q_COLLECTION_PATH"]
    retrieval_name = exp_dict["config"]['retrieval_name']

    d_collection = CollectionDatasetPreLoad(data_dir=corpus_file, id_style=id_style)
    doc_idx_to_key = d_collection.idx_to_key

    q_collection = CollectionDatasetPreLoad(data_dir=query_file, id_style=id_style)
    query_idx_to_key = q_collection.idx_to_key

    q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"], batch_size=batch_size,
                                    shuffle=False, num_workers=1)
    evaluator = SparseRetrieval(config=config, model=model, compute_stats=True, dim_voc=model.output_dim)
    retr_res = evaluator.retrieve(q_loader, top_k=exp_dict["config"]["top_k"], query_idx_to_key=query_idx_to_key,
                                name=retrieval_name, threshold=exp_dict["config"]["threshold"])

    run_path = os.path.join(config['out_dir'], f"{retrieval_name}.tsv")
    convert_ranking_to_trec(retr_res, doc_idx_to_key, run_path, tag=splade_model)



def evaluate_run(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)
    evaluate(exp_dict)

if __name__ == "__main__":
    retrieve()
    # evaluate_run()
