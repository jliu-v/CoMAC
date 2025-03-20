#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# eval_utils.py
# Author: Junfeng Liu
# Created on December 17, 2023 at 21:52

from typing import Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Precision as IgnitePrecision
from ignite.metrics import Recall as IgniteRecall


class Recall(IgniteRecall):
    def compute(self) -> Union[torch.Tensor, float]:
        try:
            return super().compute()
        except NotComputableError:
            return


class Precision(IgnitePrecision):
    def compute(self) -> Union[torch.Tensor, float]:
        try:
            return super().compute()
        except NotComputableError:
            return -1.0
