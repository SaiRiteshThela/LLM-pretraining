from typing import Iterable, Iterator, Optional
import gzip, pickle
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
RE_PAT = re.compile(PAT)

class BPETokenizer:
    __slots__ = ("vocab", "bytes2id", "merges", "rank",
                 "special_tokens", "special_to_id", "_rep", "_special_re")

    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: Optional[list[str]] = None):
        self.vocab = vocab.copy()
        self.bytes2id = {b: i for i, b in self.vocab.items()}
        self.merges = merges
        self.rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_to_id: dict[str, int] = {}
        self._rep = "\ufffd".encode("utf-8")

        next_id = (max(self.vocab) + 1) if self.vocab else 0
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            sid = self.bytes2id.get(b)
            if sid is None:
                sid = next_id
                self.vocab[sid] = b
                self.bytes2id[b] = sid
                next_id += 1
            self.special_to_id[tok] = sid

        self._special_re = (
            re.compile("|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))))
            if self.special_tokens else None
        )

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: Optional[list[str]] = None) -> "BPETokenizer":
        with gzip.open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with gzip.open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _bpe_merge(self, pre_bytes: bytes) -> list[int]:
        pieces: list[bytes] = [bytes([b]) for b in pre_bytes]
        if len(pieces) <= 1:
            return [self.bytes2id[pieces[0]]] if pieces else []

        while True:
            best_i = None
            best_rank = None
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            merged = pieces[best_i] + pieces[best_i + 1]
            pieces[best_i:best_i + 2] = [merged]

        return [self.bytes2id[p] for p in pieces]

    def _process_text_span(self, text: str) -> list[int]:
        out: list[int] = []
        for m in RE_PAT.finditer(text):                
            bt = m.group(0).encode("utf-8")
            out.extend(self._bpe_merge(bt))
        return out

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self._special_re:
            return self._process_text_span(text)

        out: list[int] = []
        last = 0
        for m in self._special_re.finditer(text):
            if m.start() > last:
                out.extend(self._process_text_span(text[last:m.start()]))
            out.append(self.special_to_id[m.group(0)])
            last = m.end()
        if last < len(text):
            out.extend(self._process_text_span(text[last:]))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for tid in self.encode(chunk):
                yield tid

    def decode(self, ids: list[int]) -> str:
        chunks = [(self.vocab.get(tid) or self._rep) for tid in ids]
        return b"".join(chunks).decode("utf-8", errors="replace")
