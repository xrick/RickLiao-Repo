

@dataclass
class Chat:
    query: str
    response: str | None

    def to_list(self) -> list[str, str]:
        return [self.query, self.response]

    def to_dict(self) -> dict[str, str]:
        return {"query": self.query, "response": self.response}

@dataclass
class ChatHistory:
    history: list[Chat]

    def __init__(self, history: list[tuple[str, str] | list[str, str]] | None = None):
        if history is None:
            self.history = []
        else:
            self.history = [Chat(*chat) for chat in history]

    def __iter__(self):
        return iter([chat.to_list() for chat in self.history])

    def __getitem__(self, index: int) -> Chat:
        return self.history[index]

    def add_chat(self, chat: Chat):
        self.history.append(chat)

    def clear_last_response(self):
        self.history[-1].response = ""

    def to_json(self) -> str:
        return json.dumps(
            [chat.to_dict() for chat in self.history], ensure_ascii=False, indent=4