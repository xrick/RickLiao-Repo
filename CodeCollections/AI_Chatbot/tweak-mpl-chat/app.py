import re
import os
import panel as pn
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from panel.io.mime_render import exec_with_return

pn.extension("codeeditor", sizing_mode="stretch_width")

LLM_MODEL = "mistral-small"
SYSTEM_MESSAGE = ChatMessage(
    role="system",
    content=(
        "You are a renowned data visualization expert "
        "with a strong background in matplotlib. "
        "Your primary goal is to assist the user "
        "in edit the code based on user request "
        "using best practices. Simply provide code "
        "in code fences (```python). You must have `fig` "
        "as the last line of code"
    ),
)

USER_CONTENT_FORMAT = """
Request:
{content}

Code:
```python
{code}
```
""".strip()

DEFAULT_MATPLOTLIB = """
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(title="Plot Title", xlabel="X Label", ylabel="Y Label")

x = np.linspace(1, 10)
y = np.sin(x)
z = np.cos(x)
c = np.log(x)

ax.plot(x, y, c="blue", label="sin")
ax.plot(x, z, c="orange", label="cos")

img = ax.scatter(x, c, c=c, label="log")
plt.colorbar(img, label="Colorbar")
plt.legend()

# must have fig at the end!
fig
""".strip()


async def callback(content: str, user: str, instance: pn.chat.ChatInterface):
    if not api_key_input.value:
        yield "Please first enter your Mistral API key"
        return
    client = MistralAsyncClient(api_key=api_key_input.value)

    # system
    messages = [SYSTEM_MESSAGE]

    # history
    messages.extend([ChatMessage(**message) for message in instance.serialize()[1:-1]])

    # new user contents
    user_content = USER_CONTENT_FORMAT.format(
        content=content, code=code_editor.value
    )
    messages.append(ChatMessage(role="user", content=user_content))

    # stream LLM tokens
    message = ""
    async for chunk in client.chat_stream(model=LLM_MODEL, messages=messages):
        if chunk.choices[0].delta.content is not None:
            message += chunk.choices[0].delta.content
            yield message

    # extract code
    llm_code = re.findall(r"```python\n(.*)\n```", message, re.DOTALL)[0]
    if llm_code.splitlines()[-1].strip() != "fig":
        llm_code += "\nfig"
    code_editor.value = llm_code


def update_plot(event):
    matplotlib_pane.object = exec_with_return(event.new)


# instantiate widgets and panes
api_key_input = pn.widgets.PasswordInput(placeholder="Enter your MistralAI API Key")
chat_interface = pn.chat.ChatInterface(
    callback=callback,
    show_clear=False,
    show_undo=False,
    show_button_name=False,
    message_params=dict(
        show_reaction_icons=False,
        show_copy_icon=False,
    ),
    height=650,
    callback_exception="verbose",
)
matplotlib_pane = pn.pane.Matplotlib(
    exec_with_return(DEFAULT_MATPLOTLIB),
    sizing_mode="stretch_both",
    tight=True,
)
code_editor = pn.widgets.CodeEditor(
    value=DEFAULT_MATPLOTLIB,
    language="python",
    sizing_mode="stretch_both",
)

# watch for code changes
code_editor.param.watch(update_plot, "value")

# lay them out
tabs = pn.Tabs(
    ("Plot", matplotlib_pane),
    ("Code", code_editor),
)

sidebar = [api_key_input, chat_interface]
main = [tabs]
template = pn.template.FastListTemplate(
    sidebar=sidebar,
    main=main,
    sidebar_width=600,
    main_layout=None,
    accent_base_color="#fd7000",
    header_background="#fd7000",
    title="Chat with Plot"
)
template.servable()
