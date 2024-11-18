r"""_summary_
-*- coding: utf-8 -*-

Module : config

File Name : __init__.py

Description : About config.
    for example:
    ```
    from config import ConfigReader
    
    config = ConfigReader.load(f, **kwargs)
    ```

Creation Date : 2024-10-30

Author : Frank Kang(frankkang@zju.edu.cn)
"""
from .reader import ConfigReader

__all__ = ['ConfigReader']
