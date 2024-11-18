#!/usr/bin/env python
r"""_summary_
-*- coding: utf-8 -*-

Module : configs.utils

File Name : utils.py

Description : utils about path

Creation Date : 2024-07-13

Author : Frank Kang(frankkang@zju.edu.cn)
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def get_dir(config_dir):
    if config_dir.startswith('.'):
        return os.path.realpath(os.path.join(ROOT, config_dir))
    else:
        return os.path.realpath(config_dir)
