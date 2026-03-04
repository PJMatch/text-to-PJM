import time
import statistics
import argparse
import spacy
import json
from datetime import datetime

DEFAULT_MODELS = [
    "pl_core_news_sm",
    "pl_core_news_md",
    "pl_core_news_lg",
    "spacy_stanza_pl",
]

DEFAULT_TEXTS = [
    "Wczoraj jadłem obiad i oglądałem film w domu.",
    "Jutro będę pracował nad projektem, a potem spotkam się ze znajomymi.",
    "Nie wiem, czy to rozwiązanie będzie działało poprawnie w każdej sytuacji.",
    "To jest test wydajności modeli spaCy dla języka polskiego.",
    "Adam i Ewa poszli do parku, gdzie spotkali swojego przyjaciela Tomka.",
    "Czy możesz mi powiedzieć, jak dojechać do najbliższej stacji metra?",
    "Kiedyś marzyłem o podróży dookoła świata, ale teraz wolę zostać w domu i czytać książki.",
    "Na stole leżały różne owoce: jabłka, pomarańcze, banany i gruszki.",
    "W zeszłym tygodniu kupiłem nowy samochód, który jest bardzo szybki i ekonomiczny.",
    "Mój kolega z pracy jest bardzo miły i zawsze chętnie pomaga innym, gdy mają problemy z komputerem.",
]

#Loading models
def load_pipeline(model_name: str):

    if model_name == "spacy_stanza_pl":
        try:
            import spacy_stanza
        except ImportError as e:
            raise OSError(
                "spacy-stanza is not installed. Please install it with 'pip install spacy-stanza' and make sure you have the Stanza models downloaded (e.g., stanza.download('pl'))."
            ) from e

        #Testing spacy-stanza on CPU torch==2.5.1+cpu" "torchvision==0.20.1+cpu" "torchaudio==2.5.1+cpu
        return spacy_stanza.load_pipeline("pl", use_gpu=False)
        return nlp

    return spacy.load(model_name)

#collecting lemma outputs for all models and texts (for comparison in the final JSON)
def collect_lemma_outputs(models: list[str], texts: list[str]) -> dict:
    lemma_outputs = {}

    for model_name in models:
        try:
            nlp = load_pipeline(model_name)
        except OSError as e:
            lemma_outputs[model_name] = {"error": str(e)}
            continue

        per_text = []
        for txt in texts:
            doc = nlp(txt)
            lemmas = [t.lemma_ for t in doc if not t.is_space and not t.is_punct]
            per_text.append({
                "original": txt,
                "lemmas_text": " ".join(lemmas),
            })

        lemma_outputs[model_name] = per_text

    return lemma_outputs

# Saving results to JSON (including lemma outputs for all models and texts)
def save_results_json(results: list[dict], path: str, args, texts: list[str]):
    if not path:
        return

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "models": args.models,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "lemmatize_only": args.lemmatize_only,
        },
        "texts_count": len(texts),
        "texts": texts,  
        "results": results,
        "lemma_outputs": collect_lemma_outputs(args.models, texts),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def bench_model(model_name: str, texts: list[str], repeats: int, warmup: int, lemmatize_only: bool):
    # Loading model
    t0 = time.perf_counter()
    try:
        nlp = load_pipeline(model_name)
    except OSError as e:
        return {"model": model_name, "error": f"Nie mogę załadować modelu: {e}"}
    load_s = time.perf_counter() - t0

    # Warmup
    for _ in range(warmup):
        for txt in texts:
            doc = nlp(txt)
            _ = [t.lemma_ for t in doc if not t.is_space]

    # Timed runs
    run_times = []
    total_tokens = 0

    for _ in range(repeats):
        start = time.perf_counter()
        for txt in texts:
            doc = nlp(txt)
            total_tokens += sum(1 for t in doc if not t.is_space)
            # Force lemma access (so we actually measure what we use)
            _ = [t.lemma_ for t in doc if not t.is_space]
        run_times.append(time.perf_counter() - start)

    mean_s = statistics.mean(run_times)
    stdev_s = statistics.pstdev(run_times) if len(run_times) > 1 else 0.0

    # tokens/sec is calculated based on the average time per single run (repeat)
    tokens_per_repeat = total_tokens / repeats if repeats else 0
    tok_per_s = (tokens_per_repeat / mean_s) if mean_s > 0 else 0

    pipe_names = ",".join(getattr(nlp, "pipe_names", [])) or "—"

    return {
        "model": model_name,
        "pipes": pipe_names,
        "load_s": load_s,
        "mean_run_s": mean_s,
        "stdev_s": stdev_s,
        "tokens_per_s": tok_per_s,
        "repeats": repeats,
        "texts": len(texts),
        "lemmatize_only": lemmatize_only,
    }


def print_results(results: list[dict]):
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]

    if failed_results:
        print("\nErrors:")
        for r in failed_results:
            print(f" - {r['model']}: {r['error']}")

    if not successful_results:
        return

    # Sort by throughput (desc)
    successful_results = sorted(successful_results, key=lambda x: x["tokens_per_s"], reverse=True)

    print("\nLOAD[s] is the one-time model loading time.")
    print("MEAN_RUN[s] is the average time to process the entire batch of texts.")
    print("TOKENS/s is the throughput calculated based on the average time.")
    print("STD[s] represents the standard deviation of run times (in seconds).")

    print("\nResults:")
    header = f"{'MODEL':<18} {'LOAD[s]':>8} {'MEAN_RUN[s]':>12} {'STD[s]':>10} {'TOKENS/s':>12}  PIPES"
    print(header)
    print("-" * len(header))
    for r in successful_results:
        print(
            f"{r['model']:<18} "
            f"{r['load_s']:>8.3f} "
            f"{r['mean_run_s']:>12.3f} "
            f"{r['stdev_s']:>10.6f} "
            f"{r['tokens_per_s']:>12.1f}  "
            f"{r['pipes']}"
        )

def show_lemma_outputs(models: list[str], texts: list[str]):
    print("\n" + "=" * 80)
    print("LEMMA OUTPUT COMPARISON")
    print("=" * 80)

    for model_name in models:
        print(f"\n### Model: {model_name}")
        print("-" * 60)

        try:
            nlp = load_pipeline(model_name)
        except OSError as e:
            print(f"Cannot load model: {e}")
            continue

        for i, txt in enumerate(texts, 1):
            doc = nlp(txt)
            lemmas = [t.lemma_ for t in doc if not t.is_space and not t.is_punct]
            print(f"\n[{i}] Original: {txt}")
            print(f"    Lemmas : {' '.join(lemmas)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="List of pipelines/models to benchmark")
    ap.add_argument("--repeats", type=int, default=20, help="Number of measurement repetitions (higher = more stable results)")
    ap.add_argument("--warmup", type=int, default=3, help="Number of warmup runs (not included in the final measurement)")
    ap.add_argument("--lemmatize-only", action="store_true", help="Disable heavy components (parser/ner) and benchmark lemmatization only")
    ap.add_argument("--save-json", type=str, default=None, help="Path to save results as JSON (e.g. results.json)")
    args = ap.parse_args()

    texts = DEFAULT_TEXTS

    results = []
    print(f"Benchmark: repeats={args.repeats}, warmup={args.warmup}, lemmatize_only={args.lemmatize_only}")
    print("Models:", ", ".join(args.models))
    print("Number of texts:", len(texts))

    for m in args.models:
        r = bench_model(m, texts, repeats=args.repeats, warmup=args.warmup, lemmatize_only=args.lemmatize_only)
        results.append(r)

    print_results(results)
    show_lemma_outputs(args.models, texts)
    if args.save_json:
        save_results_json(results, args.save_json, args, texts)
        print(f"\nSaved JSON results to: {args.save_json}")


if __name__ == "__main__":
    main()