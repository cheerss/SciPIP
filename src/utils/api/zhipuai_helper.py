r"""_summary_
-*- coding: utf-8 -*-

Module : data.utils.api.zhipuai_helper

File Name : zhipuai_helper.py

Description : Helper class for ZhipuAI interface, generally not used directly.
    For example:
    ```
    from data.utils.api import HelperCompany
    helper = HelperCompany.get()['ZhipuAI']
    ...
    ```
   
Creation Date : 2024-10-29

Author : Frank Kang(frankkang@zju.edu.cn)
"""
from zhipuai import ZhipuAI
from .base_helper import register_helper, BaseHelper


@register_helper('ZhipuAI')
class ZhipuAIHelper(BaseHelper):
    """_summary_

    Helper class for ZhipuAI interface, generally not used directly.

    For example:
    ```
    from data.utils.api import HelperCompany
    helper = HelperCompany.get()['ZhipuAI']
    ...
    ```
    """

    def __init__(self, api_key, model, base_url=None, timeout=None):
        super().__init__(api_key, model, base_url)
        self.client = ZhipuAI(api_key=api_key, base_url=base_url, timeout=timeout)
