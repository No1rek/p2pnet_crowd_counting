import yaml
import torch
from p2pnet.utils import make_path_list
from p2pnet.datasets import Dataset, DataTransformer
from p2pnet.trainer import ModelTrainer
from p2pnet.model import P2PNet
from p2pnet.criterion import Criterion


if __name__ == "__main__":
    image_folder_train = "./ShanghaiTech_part_A/train_data/images/"
    label_folder_train = "./ShanghaiTech_part_A/train_data/ground-truth/"

    with open("train_config.yaml", "r") as config_file:
        train_config = yaml.safe_load(config_file)

    train_paths, val_paths = make_path_list(image_folder_train, label_folder_train, val_fraction=0.2)
    print(f"Train: {len(train_paths)}\nVal: {len(val_paths)}")

    transformer = DataTransformer()
    train_set = Dataset(train_paths, transformer, crop=True)
    val_set = Dataset(val_paths, transformer, crop=True)

    trainer = ModelTrainer(train_config)
    model = trainer.train(P2PNet, Criterion, train_set, val_set, comment="p2pnet_train")
    torch.save(model.state_dict(), "./checkpoints/model.pt")





    
