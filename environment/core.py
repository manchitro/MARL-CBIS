import torch as th
import torch.nn as nn

from .agent import Agent
from typing import List, Tuple


def episode(agents: List[Agent], img_batch: th.Tensor, max_it: int,
            cuda: bool, eps: float, nb_class: int) -> Tuple[th.Tensor, th.Tensor]:
    """
    TODO

    :param agents:
    :type agents:
    :param img_batch:
    :type img_batch:
    :param max_it:
    :type max_it:
    :param cuda:
    :type cuda:
    :param eps:
    :type eps:
    :param nb_class:
    :type nb_class:
    :return:
    :rtype:
    """

    for a in agents:
        a.new_img(img_batch.size(0))

    for t in range(max_it):
        for a in agents:
            a.step(img_batch, eps)
        for a in agents:
            a.step_finished()

    q = th.zeros(len(agents), img_batch.size(0), nb_class,
                 device=th.device("cuda") if cuda else th.device("cpu"))

    probas = th.zeros(len(agents), img_batch.size(0),
                      device=th.device("cuda") if cuda else th.device("cpu"))

    for i, a in enumerate(agents):
        pred, proba = a.predict()
        probas[i, :] = proba
        q[i, :, :] = pred

    return nn.functional.softmax(q, dim=-1), probas


def detailled_step(agents: List[Agent], img_batch: th.Tensor, max_it: int,
                   cuda: bool, nb_class: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    TODO

    :param agents:
    :type agents:
    :param img_batch:
    :type img_batch:
    :param max_it:
    :type max_it:
    :param cuda:
    :type cuda:
    :param nb_class:
    :type nb_class:
    :return:
    :rtype:
    """
    for a in agents:
        a.new_img(img_batch.size(0))

    pos = th.zeros(max_it, len(agents), *agents[0].p[0].size(), dtype=th.long)

    q = th.zeros(max_it, len(agents), img_batch.size(0), nb_class,
                 device=th.device("cuda") if cuda else th.device("cpu"))

    probas = th.zeros(max_it, len(agents), img_batch.size(0),
                      device=th.device("cuda") if cuda else th.device("cpu"))

    for t in range(max_it):
        for i, a in enumerate(agents):
            a.step(img_batch, 0.)
            pos[t, i, :] = a.p[0]

            pred, proba = a.predict()
            probas[i, :] = proba
            q[i, :, :] = pred

        for a in agents:
            a.step_finished()

    return nn.functional.softmax(q, dim=-1), probas, pos
