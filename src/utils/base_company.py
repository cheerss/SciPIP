r"""_summary_
-*- coding: utf-8 -*-

Module : data.utils.base_company

File Name : base_company.py

Description : The base class of the factory class, used to register and reflect specific classes

Creation Date : 2024-10-29

Author : Frank Kang(frankkang@zju.edu.cn)
"""
import threading
from typing import Any
from typing_extensions import override


class BaseCompany(object):
    """_summary_

    The base class of the factory class, used to register and reflect specific classes. Use singleton mode, so it is necessary to maintain consistency in the path when importing and changing classes

    For example:
    ```
    base_company = BaseCompany.get()

    # Of course, you can also obtain the singleton using the following methods
    base_company = BaseCompany()

    entity = base_company[registered_name]
    ```
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BaseCompany, cls).__new__(
                    cls, *args, **kwargs)
                cls._instance.__init__()
        return cls._instance

    def __init__(self):
        self.entities = {}

    def init_factory(self):
        """_summary_

        Used for initializing singleton
        """
        self.entities = {}

    @classmethod
    def get(cls, *args, **kwargs):
        """_summary_

        Method for obtaining singleton classes

        For example:
        ```
        base_company = BaseCompany.get()
        entity = base_company[registered_name]
        ```

        Returns:
            BaseCompany: singleton
        """         
        if cls._instance is None:
            cls.__new__(cls, *args, **kwargs)
        return cls._instance

    def register(self, entity_name: str, entity: Any) -> bool:
        """_summary_

        Register the entity, which is called by the automatic registrar. Please do not call it yourself. Each name can only be registered once

        Args:
            entity_name (str): Name used for registration
            entity (Any): Registered entity

        Returns:
            bool: Registration success returns true, failure returns false
        """
        if entity_name not in self.entities:
            self.entities[entity_name] = entity
            return True
        else:
            return False

    def delete(self, entity_name: str) -> bool:
        """_summary_

        Remove registered entities, please use with caution

        Args:
            entity_name (str): The registered name of the registered entity

        Returns:
            bool: Success in deletion returns true, failure returns false
        """
        if entity_name in self.entities:
            self.entities[entity_name] = None
            del self.entities[entity_name]
            return True
        else:
            return False

    def __getitem__(self, key):
        return self.entities[key]

    def __len__(self):
        return len(self.entities)

    @override
    def __repr__(self) -> str:
        return "BaseCompany"
