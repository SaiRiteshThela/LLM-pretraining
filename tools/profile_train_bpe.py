import os, time, gzip, pickle, threading, psutil, cProfile, argparse
from cs336_basics.bpe import train_bpe

def measure_peak_rss_during(fn):
    import os as _os
    proc, peak, stop = psutil.Process(_os.getpid()), 0, False
    def sampler():
        nonlocal peak, stop
        while not stop:
            peak = max(peak, proc.memory_info().rss); time.sleep(0.05)
    t = threading.Thread(target=sampler, daemon=True); t.start()
    try:
        res = fn(); return res, peak
    finally:
        stop = True; t.join()

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
    OUT  = os.path.join(BASE, f"train-{fmtk(a.vocab_size)}-vocab.pkl.gz")
    os.makedirs(BASE, exist_ok=True)

    def run_training():
        with cProfile.Profile() as pr:
            res = train_bpe(INP, a.vocab_size, [a.special_token])
        pr.dump_stats(PROF); return res

    (result, peak_rss) = measure_peak_rss_during(run_training)
    vocab, merges = result
    tok_id, tok_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))

    print(f"[RSS peak] {peak_rss/1024/1024:.2f} MB")
    print(f"[profile]  {PROF}")
    print("[longest token]")
    print(f"  id        : {tok_id}")
    print(f"  byte len  : {len(tok_bytes)}")
    try: print(f"  text len  : {len(tok_bytes.decode('utf-8'))}")
    except UnicodeDecodeError: print("  text len  : n/a (non-UTF8)")
    print(f"  preview   : {safe_preview(tok_bytes)!r}")

    with gzip.open(OUT, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved]     {OUT}")

if __name__ == "__main__":
    main()
