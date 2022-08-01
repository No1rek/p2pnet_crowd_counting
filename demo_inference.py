import os
import torch
import numpy as np
from p2pnet.model import P2PNet
from p2pnet.utils import make_path_list, visualize
from p2pnet.datasets import Dataset, DataTransformer


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image_folder_test = "./ShanghaiTech_part_A/test_data/images/"
    label_folder_test = "./ShanghaiTech_part_A/test_data/ground-truth/"

    test_paths = make_path_list(image_folder_test, label_folder_test)
    transformer = DataTransformer()
    test_set = Dataset(test_paths, transformer, crop=False, keep_src=True)

    model = P2PNet(n_anchors=4, hidden_size=256, device=device)
    model.load_state_dict(torch.load("./checkpoints/model.pt"))
    model.eval()

    # compute MAE and MRSE
    MAE = 0
    RMSE = 0
    n = len(test_set)
    for i in range(n):
        features, labels = test_set[i]
        features = features.unsqueeze(0).to(device)
        coords = model.predict(features, 0.5)
        MAE += abs(len(labels) - len(coords))
        RMSE += (len(labels) - len(coords))**2
    MAE = MAE/n
    RMSE = np.sqrt(RMSE/n)
    print(f"test MAE: {MAE:.4}, MRSE: {RMSE:.4}")

    # draw predictions
    for f in os.listdir("./demo_outputs"):
        os.remove(os.path.abspath(os.path.join("./demo_outputs", f)))
    for i in np.random.choice(len(test_set), 8):
        features, labels = test_set[i]
        features = features.unsqueeze(0).to(device)
        coords = model.predict(features).detach().cpu()
        visualize(coords, test_set.view(i).permute(1,2,0), labels, f"demo_outputs/{i}.jpg")









