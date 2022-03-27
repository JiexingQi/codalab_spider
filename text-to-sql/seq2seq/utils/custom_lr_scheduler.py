from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from typing import Callable, Iterable, Optional, Tuple, Union
from transformers.file_utils import ExplicitEnum


class SchedulerType(ExplicitEnum):
    STEP_LR = "step_lr"
    MULTI_STEP_LR = "multi_step_lr"
    EXPONENTIAL_LR = "exponential_lr"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"
    # CONSTANT = "constant"
    # CONSTANT_WITH_WARMUP = "constant_with_warmup"


def get_step_lr_schedule(optimizer: Optimizer, step_size: int = 10, gamma: float = 0.5, last_epoch: int = -1):
    """
    Create a schedule that decays the learning rate of each parameter group by gamma every step_size epochs. 
    Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. 
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer) 
            – Wrapped optimizer.
        step_size (int) 
            – Period of learning rate decay.
        gamma (float) 
            – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int) 
            – The index of last epoch. Default: -1.

    Return:
        :obj:`torch.optim.lr_scheduler.StepLR` with the appropriate schedule.
    """    
    return StepLR(optimizer, step_size, gamma=gamma, last_epoch=last_epoch)


def get_multi_step_lr_schedule(optimizer: Optimizer, milestones: list = [1000, 2000, 5000, 8000], gamma: float = 0.5, last_epoch: int = -1):
    """
    Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. 
    Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. 
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer) 
            – Wrapped optimizer.
        milestones (list) 
            – List of epoch indices. Must be increasing.
        gamma (float) 
            – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int) 
            – The index of last epoch. Default: -1.

    Return:
        :obj:`torch.optim.lr_scheduler.MultiStepLR` with the appropriate schedule.
    """    
    return MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)


def get_exponential_lr_schedule(optimizer: Optimizer, gamma: float = 0.9, last_epoch: int = -1):
    """
    Decays the learning rate of each parameter group by gamma every epoch. 
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer) 
            – Wrapped optimizer.
        gamma (float) 
            – Multiplicative factor of learning rate decay.
        last_epoch (int) 
            – The index of last epoch. Default: -1.

    Return:
        :obj:`torch.optim.lr_scheduler.ExponentialLR` with the appropriate schedule.
    """    
    return ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)


def get_cosine_annealing_lr_schedule(optimizer: Optimizer, T_max: int = 1000, eta_min: int = 0, last_epoch: int = -1):
    """
    The cosine function is taken as the cycle, and the learning rate is reset at the maximum value of each cycle. 
    The initial learning rate is the maximum learning rate, and the cycle is 2*Tmax. 
    The learning rate decreases first and then increases within a cycle.

    Args:
        optimizer (Optimizer) 
            – Wrapped optimizer.
        T_max (int) 
            – Maximum number of iterations.
        eta_min (float) 
            – Minimum learning rate. Default: 0.
        last_epoch (int) 
            – The index of last epoch. Default: -1.

    Return:
        :obj:`torch.optim.lr_scheduler.ExponentialLR` with the appropriate schedule.
    """    
    return CosineAnnealingLR(optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.STEP_LR: get_step_lr_schedule,
    SchedulerType.MULTI_STEP_LR: get_multi_step_lr_schedule,
    SchedulerType.EXPONENTIAL_LR: get_exponential_lr_schedule,
    SchedulerType.COSINE_ANNEALING_LR: get_cosine_annealing_lr_schedule,
    # SchedulerType.CONSTANT: get_constant_schedule,
    # SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
}


def get_scheduler_custom(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (:obj:`str` or `:obj:`SchedulerType`):
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    return schedule_func(optimizer)
