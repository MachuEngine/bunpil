from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    """Few-shot + CoT 프롬프트 빌더. 모듈별로 인스턴스를 생성해 주입한다."""

    system: str
    few_shots: list[dict] = field(default_factory=list)
    # few_shots 형식: [{"user": "...", "assistant": "..."}]
    cot_prefix: str = ""
    # cot_prefix가 있으면 assistant 턴 앞에 붙여 CoT를 유도한다.

    def build(self, user_input: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system}]
        for shot in self.few_shots:
            messages.append({"role": "user", "content": shot["user"]})
            messages.append({"role": "assistant", "content": shot["assistant"]})
        messages.append({"role": "user", "content": user_input})
        if self.cot_prefix:
            messages.append({"role": "assistant", "content": self.cot_prefix})
        return messages
