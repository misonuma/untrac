from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_constant_schedule_with_zero(optimizer: Optimizer, num_zero_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        if current_step > num_zero_steps:
            return 1.0
        else:
            return 0.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
