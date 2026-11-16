"""Entity linking: connect KG entity names to raw text."""
from __future__ import annotations

from abc import ABC, abstractmethod

try:
    from flashtext import KeywordProcessor
except Exception:  # pragma: no cover
    KeywordProcessor = None


class EntityLinker(ABC):
    @abstractmethod
    def link(self, text: str) -> list[str]:
        """Return KG entity names mentioned in text."""
        raise NotImplementedError


class FlashTextLinker(EntityLinker):
    def __init__(self, entity_names: set[str]):
        if KeywordProcessor is None:
            raise ImportError("flashtext is required for FlashTextLinker")
        self.processor = KeywordProcessor(case_sensitive=False)
        for name in entity_names:
            clean = name.strip()
            if clean:
                self.processor.add_keyword(clean, clean)

    def link(self, text: str) -> list[str]:
        if not text:
            return []
        found = self.processor.extract_keywords(text)
        # whole-word boundary check: examine every occurrence of each keyword
        text_lower = text.lower()
        valid = []
        for name in set(found):
            name_lower = name.lower()
            start = 0
            is_valid = False
            while True:
                idx = text_lower.find(name_lower, start)
                if idx == -1:
                    break
                before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
                after_ok = (
                    idx + len(name) >= len(text_lower)
                    or not text_lower[idx + len(name)].isalnum()
                )
                if before_ok and after_ok:
                    is_valid = True
                    break
                start = idx + 1
            if is_valid:
                valid.append(name)
        return valid
