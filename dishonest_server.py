#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import flwr as fl
import torch.nn
from flwr.server.strategy import Strategy
from typing import Optional, Tuple, Dict, List
from flwr.server.strategy.aggregate import aggregate
from client import Net, DEVICE, train
from collections import OrderedDict
import numpy as np
from queue import Queue
from torch.utils.data import DataLoader, Subset
import argparse

import logging
logger = logging.getLogger(__name__)

# the index of the sample that the dishonest FC is testing for
SAMPLE_UNDER_TEST = 1

EPSILON = 1e-4
TOP_J = 5

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class DishonestStrategy(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            net: torch.nn.Module,
            sample_loader: DataLoader,
            batchsize: int,
            batches: int,
            optimizer: str,
            epsilon: float,
            top_j: int,
    ) -> None:
        """Dishonest strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, Scalar]], optionastrategy = DishonestStrategy(min_fit_clients=1,
                                 min_available_clients=1,
                                 min_eval_clients=1)
    fl.server.start_server(config={"num_rounds": 3},
                           strategy=strategy)l):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, Scalar]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters, optional): Initial global model parameters.
        """
        super().__init__()
        self.net = net
        self.epsilon = epsilon
        self.top_j = top_j
        self.batchsize = batchsize
        self.batches = batches
        self.idx: Optional[List[int]] = None
        self.norm_delta: Optional[float] = None
        self.sample_loader = sample_loader
        self.target_weights: Optional[List[np.ndarray]] = None
        self.optimizer = optimizer
        self.iout: Optional[int] = 0
        self.x0: Optional[List[torch.tensor]] = None

        self.sample = next(iter(sample_loader))

    def __repr__(self) -> str:
        rep = f"DishonestStrategy(net={self.net})"
        return rep

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return None

    def evaluate(
            self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        return None

    def prepare_weights0(self, weights):

        #for i, w in enumerate(weights):
            #logger.debug(f'{i}: {w.shape}')

        # Initialize to zero
        for i in range(4, len(weights)):
            w = weights[i]
            w.fill(0.0)
            # set biases to -1
            if i % 2 == 1:
                w.fill(-1.0)

        return weights

    def prepare_weights1(self, weights):
        # logger.debug(f'x0: {x0}')
        assert len(self.net.fcs) == 2
        iout = int(self.sample[1].detach().item())
        self.iout = iout

        for i, (x0, idx) in enumerate(zip(self.x0, self.idx)):

            # Set weights of first hidden layer
            weights[4][0 + i*2, idx] = 1.0
            weights[4][1 + i*2, idx] = -1.0
            # bias:
            weights[5][0 + i*2] = -x0
            weights[5][1 + i*2] = x0

            # 2nd hidden layer
            weights[6][0, 0 + i*2] = -1.0
            weights[6][0, 1 + i*2] = -1.0
            weights[7][0] = self.epsilon

        weights[8][iout, 0] = 1.0
        weights[9][iout] = 0.0

    def configure_fit(
            self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}

        # get weights
        weights = parameters_to_weights(parameters)

        # prepare other weights
        weights = self.prepare_weights0(weights)

        # Obtain the intermediate output after the Conv2D layers of
        # the target sample
        params_dict = zip(self.net.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        self.net.to(device=DEVICE)
        hidden = self.net.intermediate(self.sample[0].to(device=DEVICE))

        # get maximum value
        values, idx = torch.topk(torch.abs(hidden), self.top_j)
        self.idx = idx[0]
        self.x0 = [hidden[0, i0.detach().item()] for i0 in self.idx]

        self.prepare_weights1(weights)

        parameters = weights_to_parameters(weights)
        self.weights = weights
        params_dict = zip(self.net.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        # get normalized delta
        self.net.load_state_dict(state_dict, strict=True)
        train(self.net, self.sample_loader, optimizer=self.optimizer, epochs=1)

        weights_results = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        self.target_weights = weights_results
        #logger.debug(f'delta_norm: {self.weights_delta_v[0,self.imax]}')

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size = 1
        min_num_clients = 1

        if client_manager:
            if client_manager.num_available() == 0:
                logger.debug('No clients available yet')
                return list()
            elif client_manager.num_available() > 1:
                logger.error('More than one client connected')
                exit(1)
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            # Return client/config pairs
            return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # We dont need evaluation
        return []

        # # Do not configure federated evaluation if a centralized evaluation
        # # function is provided
        # if self.eval_fn is not None:
        #     return []
        #
        # # Parameters and config
        # config = {}
        # if self.on_evaluate_config_fn is not None:
        #     # Custom evaluation config function provided
        #     config = self.on_evaluate_config_fn(rnd)
        # evaluate_ins = EvaluateIns(parameters, config)
        #
        # # Sample clients
        # if rnd >= 0:
        #     sample_size, min_num_clients = self.num_evaluation_clients(
        #         client_manager.num_available()
        #     )
        #     clients = client_manager.sample(
        #         num_clients=sample_size, min_num_clients=min_num_clients
        #     )
        # else:
        #     clients = list(client_manager.all().values())
        #
        # # Return client/config pairs
        # return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if failures:
            return None, {}


        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        updated_weights = weights_results[0][0]

        delta = self._decision_statistic(updated_weights)
        logger.debug(f'delta: {delta}')

        self.norm_delta = delta

        #logger.info(f'normalized delta: {self.norm_delta}')

        return weights_to_parameters(aggregate(weights_results)), {}

    def score(self):
        return self.norm_delta

    def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        return None, {}
        # # Do not aggregate if there are failures and failures are not accepted
        # if not self.accept_failures and failures:
        #     return None, {}
        # loss_aggregated = weighted_loss_avg(
        #     [
        #         (
        #             evaluate_res.num_examples,
        #             evaluate_res.loss,
        #             evaluate_res.accuracy,
        #         )
        #         for _, evaluate_res in results
        #     ]
        # )
        # return loss_aggregated, {}

    def _decision_statistic(self, w: List[np.ndarray]):
        w_out = w[7][0]
        w0 = self.weights[7][0]
        w1 = self.target_weights[7][0]

        return self.batchsize * np.abs((w_out - w0) / (w1 - w0))


def main(sample_loader, dataset, optimizer, batchsize, batches, epsilon, top_j, port=8080, queue: Queue = None):
    net = Net(dataset=dataset)
    strategy = DishonestStrategy(net=net,
                                 batchsize=batchsize,
                                 batches=batches,
                                 optimizer=optimizer,
                                 sample_loader=sample_loader,
                                 epsilon=epsilon,
                                 top_j=top_j,
                                 )
    logger.debug(f'Server starting on port {port!s}')
    fl.server.start_server(server_address=f"[::]:{port!s}",
                           config={"num_rounds": 1},
                           strategy=strategy)
    score = strategy.score()
    if queue is not None:
        queue.put(score)
    return score
