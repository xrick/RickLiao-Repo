from typing import Dict, Generator
import re


class OPMPParser:
"""
Optimistic Progressive Markdown Parser (OPMP)
Parses partial LLM outputs into valid Markdown progressively.
"""


def __init__(self, templates: Dict[str, str]):
self.templates = templates
self.buffer = ""


def feed(self, token: str) -> str:
"""Feed a new LLM token and optimistically render Markdown."""
self.buffer += token
return self._progressive_render()


def _progressive_render(self) -> str:
"""Try to close incomplete markdown blocks progressively."""
content = self.buffer
# Auto-close code blocks and bullet lists
if content.count("```") % 2 != 0:
content += "\n```"
if re.search(r'^-\s*$', content.splitlines()[-1]):
content += "\n(…pending item…)"
return content


def finalize(self) -> str:
"""Finalize fully parsed Markdown output."""
return re.sub(r"\n{3,}", "\n\n", self.buffer.strip())