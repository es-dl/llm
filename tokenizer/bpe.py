from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


class BytePairTokenizer:
    """Byte-level BPE tokenizer with no unknown-token failure mode."""

    def __init__(self, merges=None, token_bytes=None):
        if token_bytes is None:
            token_bytes = {i: bytes([i]) for i in range(256)}

        self.token_bytes = {
            int(token_id): value if isinstance(value, bytes) else bytes(value)
            for token_id, value in token_bytes.items()
        }
        self.merges = [(int(a), int(b)) for a, b in (merges or [])]
        self._rebuild_indexes()

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    def _rebuild_indexes(self) -> None:
        self.pair_to_id = {}
        self.pair_rank = {}
        for rank, pair in enumerate(self.merges):
            token_id = 256 + rank
            self.pair_to_id[pair] = token_id
            self.pair_rank[pair] = rank

    @staticmethod
    def _text_to_chunks(text: str) -> list[bytes]:
        data = text.encode("utf-8")
        return re.findall(rb"\S+\s*|\s+", data)

    @staticmethod
    def _pair_counts(words: Counter[tuple[int, ...]]) -> Counter[tuple[int, int]]:
        counts: Counter[tuple[int, int]] = Counter()
        for tokens, freq in words.items():
            for pair in zip(tokens, tokens[1:]):
                counts[pair] += freq
        return counts

    @staticmethod
    def _merge_pair(tokens: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return tuple(merged)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        progress_every: int = 100,
    ) -> "BytePairTokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        chunks = cls._text_to_chunks(text)
        words: Counter[tuple[int, ...]] = Counter(tuple(chunk) for chunk in chunks if chunk)
        token_bytes = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[int, int]] = []

        target_merges = vocab_size - 256
        for merge_index in range(target_merges):
            pair_counts = cls._pair_counts(words)
            if not pair_counts:
                break

            pair, count = pair_counts.most_common(1)[0]
            if count < min_frequency:
                break

            new_id = 256 + merge_index
            token_bytes[new_id] = token_bytes[pair[0]] + token_bytes[pair[1]]
            next_words: Counter[tuple[int, ...]] = Counter()
            for tokens, freq in words.items():
                next_words[cls._merge_pair(tokens, pair, new_id)] += freq
            words = next_words
            merges.append(pair)

            if progress_every and (merge_index + 1) % progress_every == 0:
                print(
                    f"tokenizer merge {merge_index + 1}/{target_merges} "
                    f"pair={pair} freq={count}"
                )

        return cls(merges=merges, token_bytes=token_bytes)

    def _encode_piece(self, piece: bytes) -> list[int]:
        tokens = list(piece)
        if len(tokens) < 2 or not self.merges:
            return tokens

        while True:
            best_pair = None
            best_rank = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.pair_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            new_id = self.pair_to_id[best_pair]
            merged = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        return tokens

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._text_to_chunks(text):
            ids.extend(self._encode_piece(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.token_bytes[int(token_id)] for token_id in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "type": "byte_pair",
            "merges": self.merges,
            "token_bytes": {
                str(token_id): value.hex()
                for token_id, value in sorted(self.token_bytes.items())
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        token_bytes = {
            int(token_id): bytes.fromhex(value)
            for token_id, value in payload["token_bytes"].items()
        }
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(merges=merges, token_bytes=token_bytes)
