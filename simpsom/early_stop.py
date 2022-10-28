from copy import deepcopy
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from simpsom import SOMNet


class EarlyStop:
    """ Monitors the convergence of a map and activates
    a switch to interrupt the training if a certain tolerance
    map difference threshold is hit.

    Warning: this is a work in progress.
    Use only if you know what you are doing!
    """

    def __init__(self,
                 tolerance: float = 1e-4,
                 patience: int = 3) -> None:
        """ Initialize the early stopping class.

        Args:
            tolerance (float): the map change threshold to start
                the counter for early stopping. 
            patience (int): number of iterations with below-threshold
                map change before stopping the training.
        """

        self.tolerance = tolerance
        self.patience = patience

        self.stop_training = False
        self.convergence = []
        self.counter = 0
        self.history = None

    def calc_loss(self, net: 'SOMNet', to_monitor: str = "mapdiff") -> float:
        """ Calculate map difference convergence.

        Args:
            net (SOMNet): a SOMNet instance.
            to_monitor (str): the loss type to monitor for convergence.

        Returns:
            loss (float): the calculated loss.

        Raises:
            ValueError: if loss type is not recognized.
                Currently only map difference (mapdiff) is implemented.
        """

        all_weights = net.xp.array([n.weights for n in net.nodes_list])
        loss = None

        if self.history is not None:

            if to_monitor == "mapdiff":
                loss = net.xp.abs(net.xp.subtract(
                    all_weights, self.history)).mean()
            else:
                logger.error("Convergence method not recognized.")
                raise ValueError

        self.history = deepcopy(all_weights)

        return loss

    def check_convergence(self, loss: float) -> None:
        """ Check the change of a given loss quantity
        against its history.
        If it has been reached, activate the stop_training flag.

        Args:
            loss (float): the value to monitor.
        """

        if loss is not None:
            self.convergence.append(loss)

        if len(self.convergence) > 1 and \
                abs(self.convergence[-2] - self.convergence[-1]) < self.tolerance:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.stop_training = True
