r"""_summary_
-*- coding: utf-8 -*-

Module : prompt.pool

File Name : pool.py

Description : make prompt pool

Creation Date : 2024-11-08

Author : Frank Kang(frankkang@zju.edu.cn)
"""
import pathlib
import os
from utils.base_company import BaseCompany
from typing_extensions import Literal, override
from typing import Union, Any, IO
from .data import Prompt
from utils.path_pool import PROMPT_DIR
from glob import glob


class PromptPool(BaseCompany):
    """_summary_

    Args:
        BaseCompany (_type_): _description_
    """

    def __init__(self):
        super(PromptPool, self).__init__()
        for p in glob(os.path.join(PROMPT_DIR, '*.xml')):
            key = os.path.basename(p)[:-len('.xml')]
            prompt = Prompt(p)
            self.register(key, prompt)

    @override
    def __repr__(self) -> str:
        return "PromptPool"

    @staticmethod
    def add_prompt(file_: Union[str, pathlib.Path, IO[Any]], key: str | None = None):
        fname = ''
        if isinstance(file_, str):
            fname = file_
            if not os.path.exists(fname):
                raise FileNotFoundError(
                    'cannot find file {}'.format(fname))
        else:
            fname = file_.name

        key = os.path.basename(fname)[:-len('.xml')]
        prompt = Prompt(fname)
        PromptPool.get().register(key, prompt)
