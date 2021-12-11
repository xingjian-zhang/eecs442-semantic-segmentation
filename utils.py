import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision


def check_accuracy(loader, model, loss_fn, device='cpu'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    loss = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            loss += loss_fn(preds, y).item() * x.size(0)

            preds = torch.sigmoid(preds)
            preds = (preds >= .5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    loss = loss / len(loader.dataset)

    print(
        f'Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}\t'
        f'Dice score: {dice_score/len(loader)}\t'
        f'Loss: {loss}'
    )
    model.train()

    return num_correct / num_pixels * 100, dice_score / len(loader), loss


def get_df(dir, is_mask=False):
    """Return the data frame for CarvanaDataset."""
    car_ids = []
    paths = []
    for dir_name, _, file_names in os.walk(dir):
        for file_name in file_names:
            path = os.path.join(dir_name, file_name)
            paths.append(path)
            if is_mask:
                car_id = file_name.split('_mask')[0]
            else:
                car_id = file_name.split('.')[0]
            car_ids.append(car_id)
    if is_mask:
        col_name = "mask_path"
    else:
        col_name = "car_path"

    d = {'id': car_ids, col_name: paths}
    df_ = pd.DataFrame(data=d).set_index('id')
    return df_
