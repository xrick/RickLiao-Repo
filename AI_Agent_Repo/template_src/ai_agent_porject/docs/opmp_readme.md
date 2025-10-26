# 🧩 OPMP Integration Overview


The **Optimistic Progressive Markdown Parsing (OPMP)** system allows the agent to generate valid Markdown progressively while receiving streamed LLM tokens.


## Architecture


```
User Query → PromptService → LLM Stream → OPMP Engine → Markdown Renderer → UI
```


## Components


- **core/opmp_engine.py** — Incremental parser with auto-repair for incomplete Markdown.
- **presentation/opmp_renderer.py** — Streams progressively parsed Markdown to the UI.
- **presentation/markdown_templates/** — Jinja2 templates for structured rendering.
- **presentation/export_manager.py** — Handles Markdown/HTML/PDF exports.


## Benefits


✅ Real-time progressive rendering
✅ Error-tolerant Markdown streaming
✅ Configurable templates via YAML
✅ Seamless integration with RAG and multi-agent pipelines


## Usage Example


```python
renderer = OPMPRenderer(template_registry={})
for md in renderer.stream_markdown(["### Product A", " is great", " with high RAM"]):
print(md)
```