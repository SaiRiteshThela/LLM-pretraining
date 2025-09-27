import os, time, gzip, pickle, threading, psutil, cProfile, argparse
from cs336_basics.train_bpe import train_bpe


def measure_peak_mem_tree_during(fn, interval=0.05, prefer_pss=True):
    proc = psutil.Process(os.getpid())
    peak = 0
    stop = False

    def _mem_bytes(p):
        try:
            if prefer_pss:
                mfi = p.memory_full_info()  # exposes .pss on Linux
                pss = getattr(mfi, "pss", None)
                if pss is not None:
                    return pss
            return p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _tree_mem_bytes():
        total = _mem_bytes(proc)
        for ch in proc.children(recursive=True):
            total += _mem_bytes(ch)
        return total

    def sampler():
        nonlocal peak, stop
        while not stop:
            peak = max(peak, _tree_mem_bytes())
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        res = fn()
        return res, peak
    finally:
        stop = True
        t.join()

def safe_preview(b, n=80):
    s = b.decode("utf-8", errors="backslashreplace")
    return s if len(s) <= n else s[:n] + "…"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="TinyStoriesV2-GPT4")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--special-token", default="<|endoftext|>")
    a = p.parse_args()

    BASE = f"./artifacts/bpe/{a.dataset}/"
    INP  = f"./data/{a.dataset}-train.txt"
    fmtk = lambda n: f"{n//1000}k" if n % 1000 == 0 else str(n)
    PROF = os.path.join(BASE, f"train-{fmtk(a.vocab_size)}.prof")
    OUT_VOCAB  = os.path.join(BASE, f"train-{fmtk(a.vocab_size)}-vocab.pkl.gz")
    OUT_MERGES  = os.path.join(BASE, f"train-{fmtk(a.vocab_size)}-merges.pkl.gz")

    os.makedirs(BASE, exist_ok=True)

    def run_training():
        with cProfile.Profile() as pr:
            res = train_bpe(INP, a.vocab_size, [a.special_token])
        pr.dump_stats(PROF); return res

    (result, peak_rss) = measure_peak_mem_tree_during(run_training)
    vocab, merges = result
    tok_id, tok_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))

    print(f"[Peak RAM (tree, PSS/RSS)] {peak_rss/1024/1024:.2f} MB")
    print(f"[profile]  {PROF}")
    print("[longest token]")
    print(f"  id        : {tok_id}")
    print(f"  byte len  : {len(tok_bytes)}")
    try: print(f"  text len  : {len(tok_bytes.decode('utf-8'))}")
    except UnicodeDecodeError: print("  text len  : n/a (non-UTF8)")
    print(f"  preview   : {safe_preview(tok_bytes)!r}")

    with gzip.open(OUT_VOCAB, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved]     {OUT_VOCAB}")

    with gzip.open(OUT_MERGES, "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved]     {OUT_MERGES}")

if __name__ == "__main__":
    main()
