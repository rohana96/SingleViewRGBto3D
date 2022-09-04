"""Implements model training, evaluation and saving checkpoints"""

import time

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from SingleViewTo3D.implicit_models.occnet.models.pointTo3DNet import OccNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from SingleViewTo3D.utils import load_data, train_val_split, save_checkpoint
import config


class Train:
    """Implements model training, evaluation and saving checkpoints"""

    def __init__(self):
        """Default constructor"""

    def run(self):
        """function to train the model"""

        model = OccNet(in_size=3, features_d=64)

        if config.USE_CUDA:
            model = model.cuda()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

        points, occupancies = load_data(file_path=config.FILE_PATH)
        train_set, val_set = train_val_split(points, occupancies)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False
        )

        train_loss = []
        val_loss = []

        start_time = time.time()

        for epoch in tqdm(range(config.EPOCHS)):
            train_loss_epoch = []

            for cnt, (pts, occ) in enumerate(train_loader):

                iter_start_time = time.time()
                read_start_time = time.time()
                read_time = time.time() - read_start_time

                pred = model(torch.Tensor.float(pts))
                loss = criterion(pred, occ)
                train_loss_epoch.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_time = time.time() - start_time
                iter_time = time.time() - iter_start_time

                if (cnt % config.SAVE_FREQ) == 0:
                    save_checkpoint(epoch=epoch, model=model, model_name="occnet", optimizer=optimizer, save_dir=config.SAVE_DIR)

            train_loss_epoch = np.mean(train_loss_epoch)
            print("training loss: {}, epoch: {}".format(train_loss_epoch, epoch))
            train_loss.append(train_loss_epoch)

            with torch.no_grad():
                val_loss_epoch = []
                for val_pts, val_occ in val_loader:
                    val_pred = model(torch.Tensor.float(val_pts))
                    loss = criterion(val_pred, val_occ)
                    val_loss_epoch.append(loss.item())

                val_loss_epoch = np.mean(val_loss_epoch)
                print("validation loss: {}, epoch: {}".format(val_loss_epoch, epoch))
                val_loss.append(val_loss_epoch)

        plt.plot(range(config.EPOCHS), train_loss, 'r')
        plt.plot(range(config.EPOCHS), val_loss, 'b')
        plt.show()

    def __call__(self, *args, **kwargs):
        self.run()


def test():
    trainer = Train()
    trainer()


if __name__ == "__main__":
    test()

