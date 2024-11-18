r"""_summary_
-*- coding: utf-8 -*-

Module : prompt.utils

File Name : utils.py

Description : prompt utils

Creation Date : 2024-11-09

Author : Frank Kang(frankkang@zju.edu.cn)
"""
import sys
from .pool import PromptPool, Prompt


def get_prompt() -> Prompt:
    key = sys._getframe(1).f_code.co_name
    pool = PromptPool.get()
    if key not in pool.entities:
        raise FileNotFoundError('could not find prompt file by function name {}'.format(key))
    else:
        return pool[key]
