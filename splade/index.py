import hydra
from omegaconf import DictConfig
import os, sys
# sys.path.insert(0, os.getcwd())
# print(os.getcwd())
# print(f"In module products __package__ = {__package__}, __name__ =={__name__}")
from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing
from .utils.utils import get_initialize_config


# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def index(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)
    id_style = config.get("id_style", "row_id")
    # print(exp_dict)

    #if HF: need to udate config.
    if "hf_training" in config:
       init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
       init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None
       print('HF model')

    print(f'''data_dir={exp_dict["data"]["COLLECTION_PATH"]}, id_style={id_style}
            tokenizer_type={model_training_config["tokenizer_type"]},
            max_length={model_training_config["max_length"]},
            batch_size={config["index_retrieve_batch_size"]},''')

    model = get_model(config, init_dict)
    d_collection = CollectionDatasetPreLoad(data_dir=exp_dict["data"]["COLLECTION_PATH"], id_style=id_style)
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"],
                                    batch_size=config["index_retrieve_batch_size"],
                                    shuffle=False, num_workers=10, prefetch_factor=4)
    evaluator = SparseIndexing(model=model, config=config, compute_stats=True)
    evaluator.index(d_loader)


if __name__ == "__main__":
    index()
