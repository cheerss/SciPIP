#!/usr/bin/env python
r"""_summary_
-*- coding: utf-8 -*-

Module : prompt.data

File Name : data.py

Description : Read prompt template

Creation Date : 2024-07-16

Author : Frank Kang(frankkang@zju.edu.cn)
"""
from typing_extensions import override
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from omegaconf import DictConfig
import os


class Trunk(DictConfig):
    def __init__(self, query_node: Element) -> None:
        super(Trunk, self).__init__(content={})
        for node in query_node:
            self[node.tag] = node.text


class Query():
    def __init__(self, query_node: Element) -> None:
        super(Query, self).__init__()
        self.rank = int(query_node.get('rank'))
        self.title = query_node.find('title').text
        self.text = query_node.find('text').text
        data = query_node.find('data')
        self.data = None

        if data is not None:
            self.data = [Trunk(trunk) for trunk in data.findall('trunk')]
            if len(self.data) == 0:
                self.data = None

    @staticmethod
    def Get_Title(query_node: Element) -> str:
        return query_node.find('title').text


class AssistantCreateQuery(Query):
    TITLE = 'System Message'

    def __init__(self, query_node: Element) -> None:
        super(AssistantCreateQuery, self).__init__(query_node)

    def __call__(self, *args,
                 name=None,
                 tools=[{"type": "code_interpreter"}],
                 model="gpt-4-1106-preview",
                 **kwds) -> dict:
        """Get parameters used for client.beta.assistants.create

        Returns:
            dict: parameters used for client.beta.assistants.create
        """
        return {'role': 'system', 'content': self.text.format(*args, **kwds)} if name is None else {'name': name, 'instructions': self.text.format(*args, **kwds), 'tools': tools, 'model': model}


class MessageQuery(Query):
    TITLE = 'User Message'

    def __init__(self, query_node: Element) -> None:
        super(MessageQuery, self).__init__(query_node)

    def __call__(self, *args, **kwds) -> dict:
        """Using like str.format

        Returns:
            dict: _description_
        """
        return {'role': 'user', 'content': self.text.format(*args, **kwds)}


class Prompt(object):
    def __init__(self, path) -> None:
        """Init Prompy by xml file

        Args:
            path (_type_): _description_
        """
        super(Prompt, self).__init__()
        self.path = path
        tree = ET.parse(path)
        body = tree.getroot()
        self.queries = {}
        self.name = '.'.join(os.path.basename(path).split('.')[:-1])
        for query in body.findall('query'):
            self.__read_query__(query)

    def __read_query__(self, query_node: Element):
        title = Query.Get_Title(query_node)
        query: Query
        if title == AssistantCreateQuery.TITLE:
            query = AssistantCreateQuery(query_node)
        elif title == MessageQuery.TITLE:
            query = MessageQuery(query_node)
        else:
            raise TypeError('Title not supported!')

        if query.rank not in self.queries:
            self.queries[query.rank] = [query]
        else:
            self.queries[query.rank].append(query)

    def __getitem__(self, rank):
        return self.queries[rank]

    @override
    def __repr__(self) -> str:
        return self.name
    # def __call__(self, *args: ET.Any, **kwds: ET.Any) -> list:
