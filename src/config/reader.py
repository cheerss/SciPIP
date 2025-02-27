r"""_summary_
-*- coding: utf-8 -*-

Module : configs.reader

File Name : reader.py

Description : Load the config file, which supports referencing other configuration files. If a circular reference occurs, an exception will be thrown

Creation Date : 2024-07-13

Author : Frank Kang(frankkang@zju.edu.cn)
"""
import pathlib
import json

import os
import warnings

from typing import Union, Any, IO
from omegaconf import OmegaConf, DictConfig, ListConfig

from .utils import get_dir

class ConfigReader:
    """_summary_
    Load the config file, which supports referencing other configuration files. If a circular reference occurs, an exception will be thrown

    for examples:
    ```
    config = ConfigReader.load(file)
    ```
    """

    def __init__(
        self,
        file_: Union[str, pathlib.Path, IO[Any]],
        included: set | None = None
    ) -> None:
        """_summary_

        Args:
            file_ (Union[str, pathlib.Path, IO[Any]]): config
            included (set | None, optional): Include config file. Defaults to None.

        Raises:
            FileNotFoundError: If the configuration file cannot be found
            RecursionError: If there is a loop include
        """
        fname = ''
        self.included = included if included is not None else set()
        if isinstance(file_, str):
            fname = file_
            if not os.path.exists(fname):
                template_path = '{}.template'.format(fname)
                if os.path.exists(template_path):
                    with open(fname, 'w', encoding='utf8') as wf:
                        with open(template_path, 'r', encoding='utf8') as rf:
                            wf.write(rf.read())
                    warnings.warn(
                        'cannot find file {}. Auto generate from {}'.format(
                            fname, template_path))
                else:
                    raise FileNotFoundError(
                        'cannot find file {}'.format(fname))
        else:
            fname = file_.name

        suffix = fname.split('.')[-1]
        if suffix == 'yaml':
            config = OmegaConf.load(fname)
        elif suffix == 'json':
            if isinstance(file_, (str, IO[Any])):
                with open(file_, 'r', encoding='utf8') as f:
                    config = json.load(f)
            else:
                config = json.load(file_)
            config = DictConfig(config)
        if fname not in self.included:
            self.included.add(fname)
        else:
            raise RecursionError()
        self.__config = config
        self.complied = False

    def complie(self, config: DictConfig | None = None):
        """_summary_

        Resolve config to make include effective

        Args:
            config (DictConfig | None, optional): dict config. Defaults to None.

        Raises:
            RecursionError: If there is a loop include
        """
        modify_flag = False
        if config is None:
            config = self.__config
            modify_flag = True

        include_item = None

        for key in config.keys():
            value = config.get(key)
            if isinstance(value, DictConfig):
                self.complie(value)

        if include_item is not None:
            if isinstance(include_item, str):
                included = self.included.copy()
                if include_item in included:
                    print(include_item, included)
                    raise RecursionError()
                included.add(include_item)
                config.merge_with(ConfigReader.load(include_item, included))

            else:
                for item in include_item:
                    included = self.included.copy()
                    if item in included:
                        print(include_item, included)
                        raise RecursionError()
                    config.merge_with(ConfigReader.load(item, included))
                    included.add(item)

        if modify_flag:
            self.complied = True

    @property
    def config(self) -> DictConfig:
        """_summary_

        Obtain parsed dict config

        Returns:
            DictConfig: parsed dict config
        """
        if not self.complied:
            self.complie()
        return self.__config

    @staticmethod
    def load(
        file_: Union[str, pathlib.Path, IO[Any]],
        included: set | None = None,
        **kwargs
    ) -> DictConfig:
        """_summary_

        Class method loading configuration file

        Args:
            file_ (Union[str, pathlib.Path, IO[Any]]): config
            included (set | None, optional): Include config file. Defaults to None.

        Returns:
            DictConfig: parsed dict config
        """
        config = ConfigReader(file_, included).config
        for k, v in kwargs.items():
            config.get(k, {}).update(v)
        return config
