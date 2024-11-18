r"""_summary_
-*- coding: utf-8 -*-

Module : prompt

File Name : __init__.py

Description : About prompt.
    for example:
    ```
    from prompt import Prompt
    
    prompt = Prompt(f)
    ```

Creation Date : 2024-11-8

Author : Frank Kang(frankkang@zju.edu.cn)
"""
from .data import Prompt
from .data import AssistantCreateQuery, MessageQuery
from .pool import PromptPool

__all__ = ['Prompt', 'AssistantCreateQuery', 'MessageQuery', 'PromptPool']
