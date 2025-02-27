r"""_summary_
-*- coding: utf-8 -*-

Module : data.utils.api

File Name : __init__.py

Description : API helper automatic registration, using HelperCompany can directly reflect the corresponding helper
    If you need to add a new AI helper, please add the Python source file in the same level path, for example:
    ```
    @register_helper('name')
    class CustomerHelper(BaseHelper):
        ...
    ```
    Then import it into this file, for example
    ```
    from .customer_helper import CustomerHelper # noqa: F401, ensure autoregister
    ```

   
Creation Date : 2024-10-29

Author : Frank Kang(frankkang@zju.edu.cn)
"""

from .base_helper import HelperCompany
from .openai_helper import OpenAIHelper  # noqa: F401, ensure autoregister
from .zhipuai_helper import ZhipuAIHelper  # noqa: F401, ensure autoregister
from .local_helper import LocalHelper  # noqa: F401, ensure autoregister

__all__ = ["HelperCompany"]
