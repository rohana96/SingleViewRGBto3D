"""Implements model evaluation, visualization"""

import torch
from SingleViewTo3D.implicit_models.occnet.models.pointTo3DNet import OccNet
from torch.utils.data import DataLoader
from SingleViewTo3D.utils import load_data, train_val_split, load_model, visualize_occupancy, make_grid
import config


class Evaluate:
    """Implements model evaluation, visualization"""

    def __init__(self):
        """Default constructor"""

    def run(self, *args, **kwargs):
        """function to evaluate the model"""

        print(".....visualizing train and val data in 3D")

        points, occupancies = load_data(file_path=config.FILE_PATH)
        train_set, val_set = train_val_split(points, occupancies)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False
        )

        for loader in [train_loader, val_loader]:
            check_points, check_occs = [], []

            for pts, occs in loader:
                check_points.extend(pts)
                check_occs.extend(occs)
                if len(check_points) >= 10000:  # only visualize some points
                    break

            check_points, check_occs = torch.stack(check_points), torch.stack(check_occs)
            visualize_occupancy(check_points, check_occs)

        print(".....running inference on test data")

        resolution = config.GRID_RES  # use 128 grid points in each of the three dimensions -> 128^3 query points
        grid = make_grid(-0.5, 0.5, resolution)

        # wrap query points in data loader
        batch_size = resolution * resolution * resolution

        test_loader = torch.utils.data.DataLoader(
            grid, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False
        )

        model = OccNet(in_size=3, features_d=64)
        filepath = "{}checkpoint-occnet-epoch{}.pth".format(config.SAVE_DIR, config.EPOCHS)
        load_model(filepath=filepath, model=model)

        grid_values = None
        for pts in test_loader:
            grid_values = model(torch.Tensor.float(pts))
        grid_occupancies = grid_values > 0.  # convert model scores to classification score

        print(".....visualizing inference on test data in 3D")
        visualize_occupancy(grid, grid_occupancies)

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
