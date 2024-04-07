import torch
from model import Model
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

import etc.util.hyper_parameters as hp
from etc.util.dataset import ImdbDataset
from etc.util.datamanager import DataManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = Model().to(device)
optimizer = Adam(model.parameters(), lr=hp.lr)


def train(epochs):
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_dataloader = ImdbDataset(True).dataloader()
    test_dataloader = ImdbDataset(False).dataloader()

    for epoch in tqdm(range(epochs)):
        for idx, (input, target) in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            ascii=True,
            desc="train",
        ):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()  # 每次迭代前将上一次的梯度置零
            output = model(input)
            loss = F.nll_loss(output, target.long())
            loss.backward()  # 损失反向传播
            optimizer.step()  # 梯度更新

            train_loss_list.append(loss.cpu().item())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            train_acc_list.append(cur_acc.cpu().item())

            if idx % 100 == 0:  # 每运行 100 个 batch_size ,保存一下模型
                torch.save(model.state_dict(), "./lstm/save/imdb_model.pkl")
                torch.save(optimizer.state_dict(), "./lstm/save/imdb_optimizer.pkl")

        for idx_t, (input_t, target_t) in tqdm(enumerate(test_dataloader)):
            input_t, target_t = input_t.to(device), target_t.to(device)
            with torch.no_grad():
                output_t = model(input_t)
                # tensor.max() ([values_list], [indices_list])
                pred_t = output_t.max(dim=-1)[-1]
                # 返回的值是否与 target 相同
                cur_acc_t = pred_t.eq(target_t).float().mean()
                test_acc_list.append(cur_acc_t.cpu().item())
        print(
            "epoch: ",
            epoch,
            "train_loss: ",
            np.mean(train_loss_list),
            "train_acc: ",
            np.mean(train_acc_list),
            "test_acc",
            np.mean(test_acc_list),
        )


def test():
    loss_list = []
    acc_list = []
    model.load_state_dict(torch.load("./lstm/save/imdb_model.pkl"))
    optimizer.load_state_dict(torch.load("./lstm/save/imdb_optimizer.pkl"))
    data_loader = ImdbDataset(hp.test_data_path).dataloader()

    for idx, (input, target) in tqdm(
        enumerate(data_loader), total=len(data_loader), ascii=True, desc="test"
    ):
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu().item())
            # tensor.max() ([values_list], [indices_list])
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()  # 返回的值是否与 target 相同
            acc_list.append(cur_acc.cpu().item())
    print("total loss, acc: ", np.mean(loss_list), np.mean(acc_list))


if __name__ == "__main__":
    train(hp.epoch)
