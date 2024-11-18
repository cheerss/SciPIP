r"""_summary_
-*- coding: utf-8 -*-

Module : data.utils.api.openai_helper

File Name : openai_helper.py

Description : Helper class for openai interface, generally not used directly.
    For example:
    ```
    from data.utils.api import HelperCompany
    helper = HelperCompany.get()['OpenAI']
    ...
    ```
   
Creation Date : 2024-10-29

Author : Frank Kang(frankkang@zju.edu.cn)
"""
from openai import OpenAI
from .base_helper import register_helper, BaseHelper


@register_helper('OpenAI')
class OpenAIHelper(BaseHelper):
    """_summary_

    Helper class for openai interface, generally not used directly.

    For example:
    ```
    from data.utils.api import HelperCompany
    helper = HelperCompany.get()['OpenAI']
    ...
    ```
    """

    def __init__(self, api_key, model, base_url=None, timeout=None):
        super().__init__(api_key, model, base_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
