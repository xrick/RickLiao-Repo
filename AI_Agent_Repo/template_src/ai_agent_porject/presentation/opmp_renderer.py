from core.opmp_engine import OPMPParser


class OPMPRenderer:
def __init__(self, template_registry):
self.parser = OPMPParser(template_registry)


def stream_markdown(self, tokens: list[str]) -> Generator[str, None, None]:
"""Stream incremental Markdown as model outputs tokens."""
for token in tokens:
partial_md = self.parser.feed(token)
yield partial_md
yield self.parser.finalize()