r"""_summary_
-*- coding: utf-8 -*-

Module : data.utils.api.base_helper

File Name : base_helper.py

Description : API helper automatic registration, using HelperCompany can directly reflect the corresponding helper

Creation Date : 2024-10-29

Author : Frank Kang(frankkang@zju.edu.cn)
"""

from typing import Union, List, Optional
from abc import ABCMeta
from typing_extensions import Literal, override
from ..base_company import BaseCompany
from typing import Union
import requests
import json
from requests.exceptions import RequestException


class NotGiven:
    """
    Copy from OpenAI

    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response:
        ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


class HelperCompany(BaseCompany):
    """_summary_

    AI helper factory, inheriting BaseCompany

    For example:
    ```
    helper_company = HelperCompany.get()

    # Of course, you can also obtain the singleton using the following methods
    helper_company = HelperCompany()

    helper = helper_company[helper_name]
    ```

    @see data.utils.base_company.BaseCompany
    """

    @override
    def __repr__(self) -> str:
        return "HelperCompany"


class register_helper:
    """_summary_

    Automatically register helper annotation classes
    """

    def __init__(self, helper_type, *args, **kwds):
        self.helper_type = helper_type
        self.init_args = args
        self.init_kwds = kwds

    def __call__(self, helper_cls, *args, **kwds):
        helper_name = helper_cls.__name__
        if HelperCompany.get().register(self.helper_type, helper_cls):

            def _method(obj):
                return helper_name

            helper_cls.name = _method
            return helper_cls
        else:
            raise KeyError()


class BaseHelper:
    """_summary_

    Base class for API helper
    """

    __metaclass__ = ABCMeta

    def __init__(self, api_key, model, base_url) -> None:
        super(BaseHelper, self).__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = None

    def apply_for_service(self, data_param, max_attempts=4):
        attempt = 0
        while attempt < max_attempts:
            try:
                # print(f"尝试 #{attempt + 1}")
                r = requests.post(
                    self.base_url + "/llm/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data_param),
                )
                # 检查请求是否成功
                if r.status_code == 200:
                    # print("服务请求成功。")
                    response = r.json()["data"]["output"]
                    return response  # 或者根据需要返回其他内容
                else:
                    print("服务请求失败，响应状态码：", r.status_code)
            except RequestException as e:
                print("请求发生错误：", e)

            attempt += 1
            if attempt == max_attempts:
                print("达到最大尝试次数，服务请求失败。")
                return None  # 或者根据你的情况抛出异常

    def create(
        self,
        *args,
        messages: Union[str, List[str], List[int], object, None],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = None,
        temperature: Optional[float] | NotGiven = None,
        top_p: Optional[float] | NotGiven = None,
        max_tokens: int | NotGiven = None,
        seed: int | NotGiven = None,
        stop: Optional[Union[str, List[str], None]] | NotGiven = None,
        tools: Optional[object] | NotGiven = None,
        tool_choice: str | NotGiven = None,
        extra_headers: None | NotGiven = None,
        extra_body: None | NotGiven = None,
        timeout: float | None | NotGiven = None,
        **kwargs,
    ):
        """
        Creates a model response for the given chat conversation.

        Args:
            messages: A list of messages comprising the conversation so far.
                [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).

            stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only
                [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
                as they become available, with the stream terminated by a `data: [DONE]`
                message.
                [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

            temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
                make the output more random, while lower values like 0.2 will make it more
                focused and deterministic.

                We generally recommend altering this or `top_p` but not both.

            top_p: An alternative to sampling with temperature, called nucleus sampling, where the
                model considers the results of the tokens with top_p probability mass. So 0.1
                means only the tokens comprising the top 10% probability mass are considered.

                We generally recommend altering this or `temperature` but not both.

            max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
                completion.

                The total length of input tokens and generated tokens is limited by the model's
                context length.
                [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
                for counting tokens.

            seed: This feature is in Beta. If specified, our system will make a best effort to
                sample deterministically, such that repeated requests with the same `seed` and
                parameters should return the same result. Determinism is not guaranteed, and you
                should refer to the `system_fingerprint` response parameter to monitor changes
                in the backend.

            stop: Up to 4 sequences where the API will stop generating further tokens.

            tools: A list of tools the model may call. Currently, only functions are supported as a
                tool. Use this to provide a list of functions the model may generate JSON inputs
                for. A max of 128 functions are supported.

            tool_choice: Controls which (if any) tool is called by the model. `none` means the model will
                not call any tool and instead generates a message. `auto` means the model can
                pick between generating a message or calling one or more tools. `required` means
                the model must call one or more tools. Specifying a particular tool via
                `{"type": "function", "function": {"name": "my_function"}}` forces the model to
                call that tool.

                `none` is the default when no tools are present. `auto` is the default if tools
                are present.

            extra_headers: Send extra headers

            extra_body: Add additional JSON properties to the request

            timeout: Override the client-level default timeout for this request, in seconds
        """
        if self.model != "local":
            return (
                self.client.chat.completions.create(
                    *args,
                    model=self.model,
                    messages=messages,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    stop=stop,
                    tools=tools,
                    tool_choice=tool_choice,
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                    timeout=timeout,
                    **kwargs,
                )
                .choices[0]
                .message.content
            )
        else:
            default_system = "You are a helpful assistant."
            input_content = ""
            for message in messages:
                if message["role"] == "system":
                    default_system = message["content"]
                else:
                    input_content += message["content"]
            data_param = {}
            data_param["input"] = input_content
            data_param["serviceParams"] = {"stream": False, "system": default_system}
            data_param["ModelParams"] = {
                "temperature": 0.8,
                "presence_penalty": 2.0,
                "frequency_penalty": 0.0,
                "top_p": 0.8,
            }
            response = self.apply_for_service(data_param)
            return response
