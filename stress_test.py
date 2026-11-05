import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from test_adapter import ask_chatbot

CASES = json.loads(Path("test_cases.json").read_text(encoding="utf-8"))
QUESTIONS = [c["question"] for c in CASES]


def one_request(question):
    start = time.perf_counter()
    try:
        answer = ask_chatbot(question)
        return True, time.perf_counter() - start, str(answer)[:80]
    except Exception as exc:
        return False, time.perf_counter() - start, repr(exc)


def run_level(workers):
    questions = (QUESTIONS * ((workers // max(1, len(QUESTIONS))) + 1))[:workers]
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_request, q) for q in questions]
        for f in as_completed(futures):
            results.append(f.result())
    total = time.perf_counter() - started
    successes = sum(1 for ok, _, _ in results if ok)
    avg = sum(lat for _, lat, _ in results) / len(results)
    print(f"Concurrency={workers}: success={successes}/{len(results)}, "
          f"avg_latency={avg:.2f}s, wall_time={total:.2f}s")
    for ok, lat, msg in results:
        if not ok:
            print(f"  FAILURE after {lat:.2f}s: {msg}")


for level in [1, 3, 5]:
    run_level(level)
