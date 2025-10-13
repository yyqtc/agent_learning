def trim_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    max_tokens: int,
    token_counter: Union[
        Callable[[list[BaseMessage]], int],
        Callable[[BaseMessage], int],
        BaseLanguageModel,
    ],
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    start_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    include_system: bool = False,
    text_splitter: Optional[Union[Callable[[str], list[str]], TextSplitter]] = None,
) -> list[BaseMessage]:
    r"""将消息修剪到指定的token数量以下。

    trim_messages 可用于将聊天历史的大小减少到指定的token数量或指定的消息数量。

    无论哪种情况，如果将修剪后的聊天历史直接传递回聊天模型，生成的聊天历史通常应满足以下属性：

    1. 生成的聊天历史应该是有效的。大多数聊天模型期望聊天历史以 (1) ``HumanMessage`` 或 (2) ``SystemMessage`` 后跟 ``HumanMessage`` 开始。要实现这一点，请设置 ``start_on="human"``。
       此外，通常 ``ToolMessage`` 只能在涉及工具调用的 ``AIMessage`` 之后出现。
       有关消息的更多信息，请参阅以下链接：
       https://python.langchain.com/docs/concepts/#messages
    2. 它包含最近的消息并删除聊天历史中的旧消息。
       要实现这一点，请设置 ``strategy="last"``。
    3. 通常，如果原始聊天历史中存在 ``SystemMessage``，新的聊天历史应该包含它，因为 ``SystemMessage`` 包含对聊天模型的特殊指令。如果存在，``SystemMessage`` 几乎总是历史中的第一条消息。要实现这一点，请设置 ``include_system=True``。

    .. note::
        下面的示例展示了如何配置 ``trim_messages`` 以实现与上述属性一致的行为。

    Args:
        messages: 要修剪的类消息对象序列。
        max_tokens: 修剪后消息的最大token数量。
        token_counter: 用于计算 BaseMessage 或 BaseMessage 列表中token数量的函数或llm。如果传入 BaseLanguageModel，则将使用 BaseLanguageModel.get_num_tokens_from_messages()。
           设置为 `len` 来计算聊天历史中的**消息**数量。

            .. note::
                使用 `count_tokens_approximately` 来获得快速、近似的token计数。
                这推荐用于在热路径上使用 `trim_messages`，其中不需要精确的token计数。

        strategy: 修剪策略。

            - "first": 保留消息的前 <= n_count 个token。
            - "last": 保留消息的后 <= n_count 个token。

            默认为 ``'last'``。
        allow_partial: 如果只能包含消息的一部分，是否拆分消息。如果 ``strategy="last"`` 则包含消息的最后部分内容。如果 ``strategy="first"`` 则包含消息的第一部分内容。
           默认为 False。
        end_on: 要结束的消息类型。如果指定，则忽略此类型最后一次出现之后的每条消息。如果 ``strategy=="last"`` 则在我们尝试获取最后 ``max_tokens`` 之前完成此操作。如果 ``strategy=="first"`` 则在我们获取前 ``max_tokens`` 之后完成此操作。可以指定为字符串名称（例如 "system"、"human"、"ai"、...）或 BaseMessage 类（例如 SystemMessage、HumanMessage、AIMessage、...）。可以是单个类型或类型列表。
           默认为 None。
        start_on: 要开始的消息类型。仅在 ``strategy="last"`` 时指定。如果指定，则忽略此类型第一次出现之前的每条消息。这在我们将初始消息修剪到最后 ``max_tokens`` 之后完成。如果 ``include_system=True``，则不适用于索引 0 处的 SystemMessage。可以指定为字符串名称（例如 "system"、"human"、"ai"、...）或 BaseMessage 类（例如 SystemMessage、HumanMessage、AIMessage、...）。可以是单个类型或类型列表。
           默认为 None。
        include_system: 如果索引 0 处有 SystemMessage，是否保留它。仅在 ``strategy="last"`` 时指定。
           默认为 False。
        text_splitter: 用于拆分消息字符串内容的函数或 ``langchain_text_splitters.TextSplitter``。仅在 ``allow_partial=True`` 时使用。如果 ``strategy="last"`` 则包含部分消息的最后拆分token。如果 ``strategy=="first"`` 则包含部分消息的第一个拆分token。Token分割器假设保留分隔符，以便可以直接连接拆分内容以重新创建原始文本。默认为按换行符拆分。

    Returns:
        修剪后的 BaseMessages 列表。

    Raises:
        ValueError: 如果指定了两个不兼容的参数或未识别的 ``strategy``。

    Example:
        基于token数量修剪聊天历史，如果存在则保留 SystemMessage，并确保聊天历史以 HumanMessage（或 SystemMessage 后跟 HumanMessage）开始。

        .. code-block:: python

            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                BaseMessage,
                SystemMessage,
                trim_messages,
            )

            messages = [
                SystemMessage(
                    "you're a good assistant, you always respond with a joke."
                ),
                HumanMessage("i wonder why it's called langchain"),
                AIMessage(
                    'Well, I guess they thought "WordRope" and "SentenceString" just '
                    "didn't have the same ring to it!"
                ),
                HumanMessage("and who is harrison chasing anyways"),
                AIMessage(
                    "Hmmm let me think.\n\nWhy, he's probably chasing after the last "
                    "cup of coffee in the office!"
                ),
                HumanMessage("what do you call a speechless parrot"),
            ]


            trim_messages(
                messages,
                max_tokens=45,
                strategy="last",
                token_counter=ChatOpenAI(model="gpt-4o"),
                # 大多数聊天模型期望聊天历史以以下之一开始：
                # (1) HumanMessage 或
                # (2) SystemMessage 后跟 HumanMessage
                start_on="human",
                # 通常，如果原始历史中存在 SystemMessage，我们希望保留它。
                # SystemMessage 包含对模型的特殊指令。
                include_system=True,
                allow_partial=False,
            )

        .. code-block:: python

            [
                SystemMessage(
                    content="you're a good assistant, you always respond with a joke."
                ),
                HumanMessage(content="what do you call a speechless parrot"),
            ]

        基于消息数量修剪聊天历史，如果存在则保留 SystemMessage，并确保聊天历史以 HumanMessage（或 SystemMessage 后跟 HumanMessage）开始。

            trim_messages(
                messages,
                # 当 `len` 作为token计数器函数传入时，
                # max_tokens 将计算聊天历史中的消息数量。
                max_tokens=4,
                strategy="last",
                # 传入 `len` 作为token计数器函数将
                # 计算聊天历史中的消息数量。
                token_counter=len,
                # 大多数聊天模型期望聊天历史以以下之一开始：
                # (1) HumanMessage 或
                # (2) SystemMessage 后跟 HumanMessage
                start_on="human",
                # 通常，如果原始历史中存在 SystemMessage，我们希望保留它。
                # SystemMessage 包含对模型的特殊指令。
                include_system=True,
                allow_partial=False,
            )

        .. code-block:: python

            [
                SystemMessage(
                    content="you're a good assistant, you always respond with a joke."
                ),
                HumanMessage(content="and who is harrison chasing anyways"),
                AIMessage(
                    content="Hmmm let me think.\n\nWhy, he's probably chasing after "
                    "the last cup of coffee in the office!"
                ),
                HumanMessage(content="what do you call a speechless parrot"),
            ]


        使用自定义token计数器函数修剪聊天历史，该函数计算每条消息中的token数量。

        .. code-block:: python

            messages = [
                SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                HumanMessage(
                    "This is a 4 token text. The full message is 10 tokens.", id="first"
                ),
                AIMessage(
                    [
                        {"type": "text", "text": "This is the FIRST 4 token block."},
                        {"type": "text", "text": "This is the SECOND 4 token block."},
                    ],
                    id="second",
                ),
                HumanMessage(
                    "This is a 4 token text. The full message is 10 tokens.", id="third"
                ),
                AIMessage(
                    "This is a 4 token text. The full message is 10 tokens.",
                    id="fourth",
                ),
            ]


            def dummy_token_counter(messages: list[BaseMessage]) -> int:
                # 将每条消息视为在消息开始和结束时添加3个默认token。
                # 3 + 4 + 3 = 每条消息10个token。

                default_content_len = 4
                default_msg_prefix_len = 3
                default_msg_suffix_len = 3

                count = 0
                for msg in messages:
                    if isinstance(msg.content, str):
                        count += (
                            default_msg_prefix_len
                            + default_content_len
                            + default_msg_suffix_len
                        )
                    if isinstance(msg.content, list):
                        count += (
                            default_msg_prefix_len
                            + len(msg.content) * default_content_len
                            + default_msg_suffix_len
                        )
                return count

        前30个token，允许部分消息：
            .. code-block:: python

                trim_messages(
                    messages,
                    max_tokens=30,
                    token_counter=dummy_token_counter,
                    strategy="first",
                    allow_partial=True,
                )

            .. code-block:: python

                [
                    SystemMessage(
                        "极长的系统提示词，可能会被截断"
                    ),
                    HumanMessage(
                        "用户的第一条消息",
                    ),
                    AIMessage(
                        [{"type": "text", "text": "模型的部分回复内容"}],
                    ),
                ]

    """
    # Validate arguments
    # 检查参数兼容性：start_on 参数只能与 strategy='last' 一起使用
    if start_on and strategy == "first":
        msg = "start_on parameter is only valid with strategy='last'"
        raise ValueError(msg)
    # 检查参数兼容性：include_system 参数只能与 strategy='last' 一起使用
    if include_system and strategy == "first":
        msg = "include_system parameter is only valid with strategy='last'"
        raise ValueError(msg)

    # 将输入的消息转换为统一的 BaseMessage 格式
    messages = convert_to_messages(messages)
    
    # 处理 token_counter 参数：支持多种形式的 token 计数器
    if hasattr(token_counter, "get_num_tokens_from_messages"):
        # 如果 token_counter 是 BaseLanguageModel 实例，使用其内置的 token 计数方法
        list_token_counter = token_counter.get_num_tokens_from_messages
    elif callable(token_counter):
        # 如果 token_counter 是可调用对象，检查其参数签名
        if (
            next(iter(inspect.signature(token_counter).parameters.values())).annotation
            is BaseMessage
        ):
            # 如果函数接受单个 BaseMessage 参数，创建一个包装函数来统计多条消息
            def list_token_counter(messages: Sequence[BaseMessage]) -> int:
                return sum(token_counter(msg) for msg in messages)  # type: ignore[arg-type, misc]

        else:
            # 如果函数直接接受消息列表，直接使用
            list_token_counter = token_counter
    else:
        # 如果 token_counter 类型不支持，抛出错误
        msg = (
            f"'token_counter' expected to be a model that implements "
            f"'get_num_tokens_from_messages()' or a function. Received object of type "
            f"{type(token_counter)}."
        )
        raise ValueError(msg)

    # 处理文本分割器：用于 allow_partial=True 时分割长消息
    if _HAS_LANGCHAIN_TEXT_SPLITTERS and isinstance(text_splitter, TextSplitter):
        # 如果提供了 TextSplitter 实例，使用其 split_text 方法
        text_splitter_fn = text_splitter.split_text
    elif text_splitter:
        # 如果提供了自定义分割函数，直接使用
        text_splitter_fn = cast("Callable", text_splitter)
    else:
        # 默认使用换行符分割文本
        text_splitter_fn = _default_text_splitter

    # 根据策略选择不同的修剪实现
    if strategy == "first":
        # 使用"first"策略：保留前面的消息，丢弃后面的消息
        return _first_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            text_splitter=text_splitter_fn,
            partial_strategy="first" if allow_partial else None,
            end_on=end_on,
        )
    if strategy == "last":
        # 使用"last"策略：保留后面的消息，丢弃前面的消息（更常用）
        return _last_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            include_system=include_system,
            start_on=start_on,
            end_on=end_on,
            text_splitter=text_splitter_fn,
        )
    # 如果策略不被支持，抛出错误
    msg = f"Unrecognized {strategy=}. Supported strategies are 'last' and 'first'."
    raise ValueError(msg)