import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def tensorboard(log_path, results, step):
    writer = SummaryWriter(log_path)
    loss, val_loss = results
    writer.add_scalars("Loss/Loss", loss.items(), step)
    writer.add_scalars("Loss/Validation Loss", val_loss, step)


def run():
    Loss = [1, 2, 3]
    Val_Loss = [4, 5, 6]
    loss = np.array(Loss).mean()
    val_loss = np.array(Val_Loss).mean()

    for step in range(3):
        tensorboard("/dataset", (loss, val_loss), step)


if __name__ == "__main__":
    run()
