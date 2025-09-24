from typing import Optional, Iterator
from multiprocessing import cpu_count, get_context
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict 

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _doc_split_pat(special_tokens: Optional[list[str]]) -> str:
    special_tokens = [tok for tok in special_tokens if tok] if special_tokens else []
    return re.compile('|'.join(sorted(map(re.escape, special_tokens), key=len, reverse=True)))


def _doc_itr(data: str, doc_split_pat: str) -> Iterator[str]:
    if not doc_split_pat:
        yield data
        return
    
    regex = re.compile(doc_split_pat)
    last = 0
    for m in regex.finditer(data):
        if m.start() > last:
            yield data[last:m.start()]
        last = m.end()

    if last < len(data):
        yield data[last:]

    return


def _process_chunks(input_path: str, start: int, end:int, doc_split_pat: str) -> Counter[tuple[int, ...]]:
    word_count: Counter[tuple[int, ...]] = Counter()
    prep_regex = re.compile(PAT)
    with open(input_path, 'rb') as f:
        f.seek(start)
        data = f.read(end-start).decode('utf-8', errors='ignore')
        for text in _doc_itr(data, doc_split_pat=doc_split_pat):
            for m in prep_regex.finditer(text):
                word_count[tuple(text[m.start():m.end()].encode('utf-8'))] += 1

    return word_count

def _get_pair_cnt_map(word_cnt: Counter[tuple[int, ...]]) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[tuple[int, ...]]]]:
    pair_cnt: Counter[tuple[int, int]] = Counter()
    pair_map: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for w, f in word_cnt.items():
        if len(w) > 1:
            for l, r in zip(w[:-1], w[1:]):
                pair_cnt[tuple([l, r])] += f
                pair_map[tuple([l, r])].add(w)

    return pair_cnt, pair_map

def _get_pair_from_word(word: str) -> Counter[tuple[int, int]]:
    word_cnt: Counter[tuple[int, int]] = Counter()
    if len(word) < 2:
        return word_cnt
    
    for l, r in zip(word[:-1], word[1:]):
        word_cnt[tuple([l, r])] += 1

    return word_cnt

def _merge_word(word: tuple[int, ...], pair: tuple[int, int], tok_id) -> tuple[int, ...]:
    new_word: list[int] = []
    if len(word) < 2:
        return word
    
    itr = 0
    while itr < len(word)-1:
        if word[itr] == pair[0] and word[itr+1] == pair[1]:
            new_word.append(tok_id)
            itr+=2
        else:
            new_word.append(word[itr])
            itr+=1

    if itr != len(word):
        new_word.append(word[itr])

    return tuple(new_word)


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: Optional[list[str]] = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {i:bytes([i]) for i in range(256)}
    special_tokens = [tok for tok in special_tokens if tok] if special_tokens else []
    curr_tok_idx = 256

    for tok in special_tokens:
        vocab[curr_tok_idx] = tok.encode('utf-8')
        curr_tok_idx += 1

    merges: list[tuple[bytes, bytes]] = []

    if vocab_size <= len(vocab):
        return vocab, merges

    num_processes = max(1, min(cpu_count(), 4))
    doc_pat = _doc_split_pat(special_tokens=special_tokens)
    with open(input_path, 'rb') as f:
        bounds = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')
    tasks = [(input_path, s, e, doc_pat) for s, e in zip(bounds[:-1], bounds[1:])]

    word_cnt: Counter[tuple[int, ...]] = Counter()
    if num_processes > 1:
        ctx = get_context('fork')
        with ctx.Pool(processes=num_processes) as p:
            word_cnt_dicts = p.starmap(_process_chunks, tasks)
    else:
        word_cnt_dicts = [_process_chunks(*t) for t in tasks]

    for dct in word_cnt_dicts:
        word_cnt.update(dct)

    pair_cnt, pair_map = _get_pair_cnt_map(word_cnt)

    while curr_tok_idx < vocab_size:
        best_pair, _ = max(pair_cnt.items(), key=lambda kv: (kv[1], vocab[kv[0][0]], vocab[kv[0][1]]))
        reduced_list = pair_map.get(best_pair)
        vocab[curr_tok_idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append(tuple([vocab[best_pair[0]], vocab[best_pair[1]]]))
        for w in list(reduced_list):
            freq = word_cnt[w]
            old_pairs = _get_pair_from_word(w)
            new_word = _merge_word(w, best_pair, curr_tok_idx)
            new_pairs = _get_pair_from_word(new_word)

            for pair, c in old_pairs.items():
                pair_cnt[pair] -= freq * c
                pair_map[pair].discard(w)

            for pair, c in new_pairs.items():
                pair_cnt[pair] += freq * c
                pair_map[pair].add(new_word)

            word_cnt.pop(w, None)
            word_cnt[new_word] = freq
        curr_tok_idx += 1
    return vocab, merges