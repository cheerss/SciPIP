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
   
Creation Date : 2024-11-28

Author : lihuigu(lihuigu@zju.edu.cn)
"""

from .base_helper import register_helper, BaseHelper


@register_helper("Local")
class LocalHelper(BaseHelper):
    """_summary_

    Helper class for ZhipuAI interface, generally not used directly.

    For example:
    ```
    from data.utils.api import HelperCompany
    helper = HelperCompany.get()['Local']
    ...
    ```
    """

    def __init__(self, api_key, model, base_url=None, timeout=None):
        super().__init__(api_key, model, base_url)
