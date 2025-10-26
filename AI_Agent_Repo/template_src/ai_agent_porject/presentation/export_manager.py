import markdown
from pathlib import Path


class ExportManager:
def export_markdown(self, content: str, output_path: str):
Path(output_path).write_text(content, encoding='utf-8')


def export_html(self, content: str, output_path: str):
html = markdown.markdown(content)
Path(output_path).write_text(html, encoding='utf-8')