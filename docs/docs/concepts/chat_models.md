# 聊天模型

## 概述

大型语言模型（LLMs）是先进的机器学习模型，擅长处理各种语言相关的任务，如文本生成、翻译、摘要、问答等，而无需针对每种场景进行特定的微调。

现代LLMs通常通过聊天模型接口访问，该接口接受一系列[消息](/docs/concepts/messages)作为输入，并返回一条[消息](/docs/concepts/messages)作为输出。

最新一代聊天模型提供了额外的功能：

* [工具调用](/docs/concepts/tool_calling)：许多流行的聊天模型提供了原生的[工具调用](/docs/concepts/tool_calling) API。该API允许开发人员构建丰富的应用程序，使LLMs能够与外部服务、API和数据库进行交互。工具调用还可用于从非结构化数据中提取结构化信息并执行各种其他任务。
* [结构化输出](/docs/concepts/structured_outputs)：一种使聊天模型以结构化格式（如与给定模式匹配的JSON）响应的技术。
* [多模态](/docs/concepts/multimodality)：处理文本以外的数据的能力；例如，图像、音频和视频。

## 特性

LangChain提供了一致的接口，用于与来自不同提供者的聊天模型进行交互，同时提供额外的功能，用于监控、调试和优化使用LLMs的应用程序的性能。

* 与许多聊天模型提供者的集成（例如，Anthropic、OpenAI、Ollama、Microsoft Azure、Google Vertex、Amazon Bedrock、Hugging Face、Cohere、Groq）。请参阅[聊天模型集成](/docs/integrations/chat/)以获取最新的支持模型列表。
* 使用LangChain的[消息](/docs/concepts/messages)格式或OpenAI格式。
* 标准[工具调用API](/docs/concepts/tool_calling)：用于将工具绑定到模型、访问模型发出的工具调用请求并将工具结果发送回模型的标准接口。
* 通过`with_structured_output`方法提供标准的[结构化输出](/docs/concepts/structured_outputs/#structured-output-method) API。
* 支持[异步编程](/docs/concepts/async)、[高效批处理](/docs/concepts/runnables/#optimized-parallel-execution-batch)、[丰富的流式API](/docs/concepts/streaming)。
* 与[LangSmith](https://docs.smith.langchain.com)集成，用于监控和调试基于LLMs的生产级应用程序。
* 其他功能，如标准化的[令牌使用](/docs/concepts/messages/#aimessage)、[速率限制](#rate-limiting)、[缓存](#caching)等。

## 集成

LangChain有许多聊天模型集成，允许您使用来自不同提供者的各种模型。

这些集成分为两种类型：

1. **官方模型**：这些模型是LangChain和/或模型提供者正式支持的模型。您可以在`langchain-<provider>`包中找到这些模型。
2. **社区模型**：这些模型主要由社区贡献和支持。您可以在`langchain-community`包中找到这些模型。

LangChain聊天模型的命名约定是在类名之前加上"Chat"前缀（例如，`ChatOllama`、`ChatAnthropic`、`ChatOpenAI`等）。

请查看[聊天模型集成](/docs/integrations/chat/)以获取支持的模型列表。

:::note
不包含"Chat"前缀或以"LLM"作为后缀的模型通常指的是不遵循聊天模型接口的旧模型，而是使用接受字符串作为输入并返回字符串作为输出的接口。
:::

## 接口

LangChain聊天模型实现了[BaseChatModel](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html)接口。因为`BaseChatModel`也实现了[可运行接口](/docs/concepts/runnables)，聊天模型支持[标准流式接口](/docs/concepts/streaming)、[异步编程](/docs/concepts/async)、优化的[批处理](/docs/concepts/runnables/#optimized-parallel-execution-batch)等。有关可运行接口的更多详细信息，请参阅[可运行接口](/docs/concepts/runnables)。

聊天模型的许多关键方法以[消息](/docs/concepts/messages)作为输入并返回消息作为输出。

聊天模型提供了一组标准参数，可用于配置模型。这些参数通常用于控制模型的行为，例如输出的温度、响应中的最大令牌数以及等待响应的最大时间。有关标准参数的更多详细信息，请参阅[标准参数](#standard-parameters)部分。

:::note
在文档中，我们通常将"LLM"和"聊天模型"交替使用。这是因为大多数现代LLMs通过聊天模型接口向用户暴露。

然而，LangChain也实现了不遵循聊天模型接口的旧LLMs，而是使用接受字符串作为输入并返回字符串作为输出的接口。这些模型通常没有"Chat"前缀（例如，`Ollama`、`Anthropic`、`OpenAI`等）。
这些模型实现了[BaseLLM](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html#langchain_core.language_models.llms.BaseLLM)接口，并可能以"LLM"后缀命名（例如，`OllamaLLM`、`AnthropicLLM`、`OpenAILLM`等）。通常，用户不应使用这些模型。
:::

### 关键方法

聊天模型的关键方法包括：

1. **invoke**：与聊天模型交互的主要方法。它接受一系列[消息](/docs/concepts/messages)作为输入，并返回一系列消息作为输出。
2. **stream**：允许您在生成时流式传输聊天模型输出的方法。
3. **batch**：允许您将多个请求批量处理以提高效率的方法。
4. **bind_tools**：允许您将工具绑定到聊天模型以便在模型的执行上下文中使用的方法。
5. **with_structured_output**：一个包装器，用于支持[结构化输出](/docs/concepts/structured_outputs)的模型的`invoke`方法。

其他重要方法可以在[BaseChatModel API参考](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html)中找到。

### 输入和输出

现代LLMs通常通过聊天模型接口访问，该接口接受[消息](/docs/concepts/messages)作为输入并返回[消息](/docs/concepts/messages)作为输出。消息通常与角色（例如，"系统"、"人类"、"助手"）相关联，并包含一个或多个内容块，这些内容块包含文本或潜在的多模态数据（例如，图像、音频、视频）。

LangChain支持两种消息格式与聊天模型进行交互：

1. **LangChain消息格式**：LangChain自己的消息格式，默认使用，并在LangChain内部使用。
2. **OpenAI消息格式**：OpenAI的消息格式。

### 标准参数

许多聊天模型具有标准化的参数，可用于配置模型：

| 参数          | 描述                                                                                                                                                                                                                                                                                                    |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`        | 您想要使用的特定AI模型的名称或标识符（例如，"gpt-3.5-turbo"或"gpt-4"）。                                                                                                                                                                                                        |
| `temperature`  | 控制模型输出的随机性。较高的值（例如，1.0）使响应更具创造性，而较低的值（例如，0.0）使其更具确定性和专注性。                                                                                                                            |
| `timeout`      | 等待模型响应的最大时间（以秒为单位），在超时之前取消请求。确保请求不会无限期挂起。                                                                                                                                                               |
| `max_tokens`   | 限制响应中的令牌总数（单词和标点符号）。这控制输出的长度。                                                                                                                                                                                           |
| `stop`         | 指定停止序列，指示模型何时应停止生成令牌。例如，您可以使用特定字符串来表示响应的结束。                                                                                                                                              |
| `max_retries`  | 系统在由于网络超时或速率限制等问题而失败时，重新发送请求的最大尝试次数。                                                                                                                                                                        |
| `api_key`      | 进行模型提供者身份验证所需的API密钥。通常在您注册访问模型时发放。                                                                                                                                                                              |
| `base_url`     | 发送请求的API端点的URL。通常由模型提供者提供，并且对于定向请求是必要的。                                                                                                                                                          |
| `rate_limiter` | 可选的[BaseRateLimiter](https://python.langchain.com/api_reference/core/rate_limiters/langchain_core.rate_limiters.BaseRateLimiter.html#langchain_core.rate_limiters.BaseRateLimiter)，用于间隔请求以避免超过速率限制。有关更多详细信息，请参见[速率限制](#rate-limiting)。 |

一些重要的注意事项：

- 标准参数仅适用于公开具有预期功能的参数的模型提供者。例如，某些提供者不公开最大输出令牌的配置，因此max_tokens无法得到支持。
- 标准参数目前仅在具有自己集成包的集成中强制执行（例如`langchain-openai`、`langchain-anthropic`等），而不在`langchain-community`中的模型上强制执行。

聊天模型还接受其他特定于该集成的参数。要查找聊天模型支持的所有参数，请访问其各自的[API参考](https://python.langchain.com/api_reference/)。

## 工具调用

聊天模型可以调用[工具](/docs/concepts/tools)来执行任务，例如从数据库中获取数据、进行API请求或运行自定义代码。有关更多信息，请参见[工具调用](/docs/concepts/tool_calling)指南。

## 结构化输出

可以请求聊天模型以特定格式（例如JSON或匹配特定模式）进行响应。此功能对于信息提取任务非常有用。有关该技术的更多信息，请阅读[结构化输出](/docs/concepts/structured_outputs)指南。

## 多模态

大型语言模型（LLMs）不仅限于处理文本。它们还可以用于处理其他类型的数据，例如图像、音频和视频。这被称为[多模态](/docs/concepts/multimodality)。

目前，只有一些LLMs支持多模态输入，几乎没有支持多模态输出。有关详细信息，请查阅特定模型文档。

## 上下文窗口

聊天模型的上下文窗口指的是模型一次可以处理的最大输入序列大小。虽然现代LLMs的上下文窗口相当大，但在与聊天模型交互时，开发人员必须牢记这一限制。

如果输入超过上下文窗口，模型可能无法处理整个输入，并可能引发错误。在对话应用程序中，这一点尤其重要，因为上下文窗口决定了模型在整个对话中可以"记住"多少信息。开发人员通常需要管理输入，以保持对话的连贯性而不超过限制。有关在对话中处理记忆的更多详细信息，请参阅[记忆](https://langchain-ai.github.io/langgraph/concepts/memory/)。

输入的大小以[令牌](/docs/concepts/tokens)为单位进行测量，这是模型使用的处理单位。

## 高级主题

### 速率限制

许多聊天模型提供者对在给定时间段内可以发出的请求数量施加限制。

如果您达到速率限制，通常会收到提供者的速率限制错误响应，并需要等待才能发出更多请求。

您有几种选择来处理速率限制：

1. 尝试通过间隔请求来避免达到速率限制：聊天模型接受在初始化时提供的`rate_limiter`参数。该参数用于控制向模型提供者发出的请求的速率。在基准测试模型以评估其性能时，间隔请求是一种特别有用的策略。有关如何使用此功能的更多信息，请参见[如何处理速率限制](/docs/how_to/chat_model_rate_limiting/)。
2. 尝试从速率限制错误中恢复：如果您收到速率限制错误，可以在重试请求之前等待一段时间。等待的时间可以随着每个后续的速率限制错误而增加。聊天模型具有`max_retries`参数，可用于控制重试次数。有关更多详细信息，请参见[标准参数](#standard-parameters)部分。
3. 回退到另一个聊天模型：如果您在一个聊天模型上达到速率限制，可以切换到另一个未受限的聊天模型。

### 缓存

聊天模型API可能很慢，因此一个自然的问题是是否要缓存先前对话的结果。理论上，缓存可以通过减少对模型提供者的请求数量来提高性能。

然而，实际上，缓存聊天模型响应是一个复杂的问题，应谨慎处理。

原因是，如果依赖于缓存**确切**的输入，获取缓存命中的可能性在对话的第一次或第二次交互后不太可能。例如，多个对话开始时完全相同的消息的可能性有多大？那三条消息完全相同的可能性呢？

一种替代方法是使用语义缓存，根据输入的含义而不是输入本身的确切内容来缓存响应。这在某些情况下可能有效，但在其他情况下则不然。

语义缓存在应用程序的关键路径上引入了对另一个模型的依赖（例如，语义缓存可能依赖于[嵌入模型](/docs/concepts/embedding_models)将文本转换为向量表示），并且不能保证准确捕捉输入的含义。

然而，在某些情况下，缓存聊天模型响应可能是有益的。例如，如果您有一个聊天模型用于回答常见问题，缓存响应可以帮助减少对模型提供者的负载、成本并提高响应时间。

有关更多详细信息，请参见[如何缓存聊天模型响应](/docs/how_to/chat_model_caching/)指南。

## 相关资源

* 使用聊天模型的操作指南：[操作指南](/docs/how_to/#chat-models)。
* 支持的聊天模型列表：[聊天模型集成](/docs/integrations/chat/)。

### 概念指南

* [消息](/docs/concepts/messages)
* [工具调用](/docs/concepts/tool_calling)
* [多模态](/docs/concepts/multimodality)
* [结构化输出](/docs/concepts/structured_outputs)
* [令牌](/docs/concepts/tokens)
