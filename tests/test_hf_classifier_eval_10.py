import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.classifiers.parser import parse_classification
from src.classifiers.prompts import SYSTEM_PROMPT
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION
from src.storage.minio_client import MinIOStorage


MODEL = "Qwen/Qwen3-32B"
PROVIDER = "nscale"

EVAL_FILE = Path("tests/data/pib_classifier_eval_20.json")
OUTPUT_FILE = Path("tests/data/qwen3_32b_hf_eval_results.json")
CSV_FILE = Path("tests/data/qwen3_32b_hf_eval_results.csv")

# Current Nscale listing for Qwen3-32B:
# $0.08 / 1M input tokens
# $0.25 / 1M output tokens
INPUT_PRICE_PER_TOKEN = 0.08 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 0.25 / 1_000_000


def build_user_prompt(article):
    chapters = "\n".join(
        f"{number}. {title}"
        for number, title in LAXMIKANTH_8TH_EDITION.items()
    )

    return f"""
Classify the following PIB article.

You must first determine whether the article contains substantive
Indian Polity content.

If irrelevant:
- relevant = false
- chapter_number = null
- topic = null
- subtopic = null

If relevant:
- relevant = true
- select exactly one Laxmikanth chapter
- provide the most appropriate topic
- provide a subtopic when possible

Keep the reason concise: no more than 25 words.
The reason should state the main basis for the classification only.

Do not classify an article as relevant merely because:
- a minister is mentioned
- a government department is mentioned
- a government scheme is mentioned
- government funding is mentioned
- the article is issued by PIB

ARTICLE TITLE:
{article["title"]}

ARTICLE:
{article["article_text"]}

LAXMIKANTH CHAPTERS:
{chapters}

Return JSON only.
"""


RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "polity_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {"type": "boolean"},
                "chapter_number": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 92,
                },
                "topic": {"type": ["string", "null"]},
                "subtopic": {"type": ["string", "null"]},
                "reason": {"type": "string", "maxLength": 300},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "relevant",
                "chapter_number",
                "topic",
                "subtopic",
                "reason",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
}


def main():
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be set in .env")

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        evaluation_set = json.load(f)

    client = InferenceClient(
        provider=PROVIDER,
        api_key=token,
    )

    storage = MinIOStorage()

    objects = storage.list_objects("pib/raw/")
    article_objects = {
        Path(obj.object_name).stem: obj.object_name
        for obj in objects
    }

    results = []

    print(f"Model: {MODEL}")
    print(f"Provider: {PROVIDER}")
    print(f"Articles: {len(evaluation_set)}")
    print("=" * 80)

    for index, article in enumerate(evaluation_set, start=1):
        prid = article["prid"]

        print(
            f"[{index}/{len(evaluation_set)}] "
            f"PRID {prid} - {article['title']}"
        )

        object_name = article_objects.get(prid)

        if object_name is None:
            print(f"    ERROR: MinIO object not found for PRID {prid}")
            results.append({
                "prid": prid,
                "title": article["title"],
                "gold_label": article["gold_label"],
                "gold_chapter": article["gold_chapter"],
                "model_relevant": None,
                "model_chapter": None,
                "reason": None,
                "confidence": None,
                "error": f"MinIO object not found for PRID {prid}",
                "runtime_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "input_cost_usd": None,
                "output_cost_usd": None,
                "total_cost_usd": None,
                "raw_response": None,
            })
            continue

        pib_record = storage.get_json(object_name)

        article_for_model = {
            **article,
            "article_text": pib_record["article_text"],
        }

        prompt = build_user_prompt(article_for_model)

        start = time.perf_counter()

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format=RESPONSE_SCHEMA,
                temperature=0,
                max_tokens=1024,
            )

            elapsed = time.perf_counter() - start

            raw_response = completion.choices[0].message.content

            usage = completion.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            input_cost = input_tokens * INPUT_PRICE_PER_TOKEN
            output_cost = output_tokens * OUTPUT_PRICE_PER_TOKEN
            total_cost = input_cost + output_cost

            try:
                parsed = parse_classification(raw_response)

                model_relevant = parsed.relevant
                model_chapter = parsed.chapter_number
                reason = parsed.reason
                confidence = parsed.confidence
                error = None

            except Exception as exc:
                model_relevant = None
                model_chapter = None
                reason = None
                confidence = None
                error = str(exc)

        except Exception as exc:
            elapsed = time.perf_counter() - start
            raw_response = None
            input_tokens = None
            output_tokens = None
            total_tokens = None
            input_cost = None
            output_cost = None
            total_cost = None
            model_relevant = None
            model_chapter = None
            reason = None
            confidence = None
            error = f"HF request failed: {exc}"

        result = {
            "prid": prid,
            "title": article["title"],
            "gold_label": article["gold_label"],
            "gold_chapter": article["gold_chapter"],
            "model_relevant": model_relevant,
            "model_chapter": model_chapter,
            "reason": reason,
            "confidence": confidence,
            "error": error,
            "runtime_seconds": round(elapsed, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": (
                round(input_cost, 10) if input_cost is not None else None
            ),
            "output_cost_usd": (
                round(output_cost, 10) if output_cost is not None else None
            ),
            "total_cost_usd": (
                round(total_cost, 10) if total_cost is not None else None
            ),
            "raw_response": raw_response,
        }

        results.append(result)

        print(
            f"    Gold:   {article['gold_label']} / {article['gold_chapter']}"
        )
        print(
            f"    Model:  {model_relevant} / {model_chapter}"
        )
        print(
            f"    Tokens: {input_tokens} in / "
            f"{output_tokens} out / {total_tokens} total"
        )
        print(
            f"    Cost:   ${total_cost:.8f}"
            if total_cost is not None
            else "    Cost:   unavailable"
        )
        print(f"    Time:   {elapsed:.2f}s")
        if error:
            print(f"    Error:  {error}")
        print()

    # Aggregate metrics
    valid_results = [
        r for r in results
        if r["error"] is None
        and r["model_relevant"] is not None
    ]

    clear_results = [
        r for r in valid_results
        if r["gold_label"] in ("relevant", "irrelevant")
    ]

    clear_relevance_correct = sum(
        (
            r["model_relevant"] is True
            if r["gold_label"] == "relevant"
            else r["model_relevant"] is False
        )
        for r in clear_results
    )

    clear_relevant = [
        r for r in valid_results
        if r["gold_label"] == "relevant"
    ]

    clear_chapter_correct = sum(
        r["model_relevant"] is True
        and r["model_chapter"] == r["gold_chapter"]
        for r in clear_relevant
    )

    total_input = sum(
        r["input_tokens"] for r in results
        if r["input_tokens"] is not None
    )
    total_output = sum(
        r["output_tokens"] for r in results
        if r["output_tokens"] is not None
    )
    total_tokens = sum(
        r["total_tokens"] for r in results
        if r["total_tokens"] is not None
    )
    total_cost = sum(
        r["total_cost_usd"] for r in results
        if r["total_cost_usd"] is not None
    )
    total_runtime = sum(
        r["runtime_seconds"] for r in results
        if r["runtime_seconds"] is not None
    )

    successful_requests = sum(
        r["total_tokens"] is not None for r in results
    )

    metrics = {
        "model": MODEL,
        "provider": PROVIDER,
        "articles_requested": len(evaluation_set),
        "successful_requests": successful_requests,
        "valid_classifications": len(valid_results),
        "clear_relevance_correct": clear_relevance_correct,
        "clear_relevance_total": len(clear_results),
        "clear_relevance_accuracy": (
            clear_relevance_correct / len(clear_results)
            if clear_results else None
        ),
        "clear_relevant_chapter_correct": clear_chapter_correct,
        "clear_relevant_chapter_total": len(clear_relevant),
        "clear_relevant_chapter_accuracy": (
            clear_chapter_correct / len(clear_relevant)
            if clear_relevant else None
        ),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "average_input_tokens_per_article": (
            total_input / successful_requests
            if successful_requests else None
        ),
        "average_output_tokens_per_article": (
            total_output / successful_requests
            if successful_requests else None
        ),
        "average_total_tokens_per_article": (
            total_tokens / successful_requests
            if successful_requests else None
        ),
        "total_cost_usd": total_cost,
        "average_cost_usd_per_article": (
            total_cost / successful_requests
            if successful_requests else None
        ),
        "total_runtime_seconds": total_runtime,
        "average_runtime_seconds": (
            total_runtime / successful_requests
            if successful_requests else None
        ),
        "estimated_cost_15_articles_per_day_usd": (
            (total_cost / successful_requests) * 15
            if successful_requests else None
        ),
        "estimated_cost_20_articles_per_day_usd": (
            (total_cost / successful_requests) * 20
            if successful_requests else None
        ),
        "estimated_cost_20_articles_per_30_days_usd": (
            (total_cost / successful_requests) * 20 * 30
            if successful_requests else None
        ),
        "estimated_cost_20_articles_per_year_usd": (
            (total_cost / successful_requests) * 20 * 365
            if successful_requests else None
        ),
    }

    output = {
        "metrics": metrics,
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    fieldnames = list(results[0].keys()) if results else []

    with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Valid classifications: {len(valid_results)}/{len(evaluation_set)}")
    print(
        f"Clear relevance accuracy: "
        f"{clear_relevance_correct}/{len(clear_results)}"
        if clear_results
        else "Clear relevance accuracy: N/A"
    )
    print(
        f"Clear relevant chapter accuracy: "
        f"{clear_chapter_correct}/{len(clear_relevant)}"
        if clear_relevant
        else "Clear relevant chapter accuracy: N/A"
    )
    print(f"Total input tokens:  {total_input:,}")
    print(f"Total output tokens: {total_output:,}")
    print(f"Total tokens:        {total_tokens:,}")
    print(f"Average tokens/article: {total_tokens / successful_requests:.1f}"
          if successful_requests else "Average tokens/article: N/A")
    print(f"Total estimated cost: ${total_cost:.8f}")
    print(
        f"Average cost/article: "
        f"${total_cost / successful_requests:.8f}"
        if successful_requests
        else "Average cost/article: N/A"
    )
    print(
        f"Estimated 20/day for 30 days: "
        f"${(total_cost / successful_requests) * 20 * 30:.6f}"
        if successful_requests
        else "Estimated 20/day for 30 days: N/A"
    )
    print(
        f"Estimated 20/day for 1 year: "
        f"${(total_cost / successful_requests) * 20 * 365:.6f}"
        if successful_requests
        else "Estimated 20/day for 1 year: N/A"
    )
    print(f"Total runtime: {total_runtime:.2f}s")
    print(
        f"Average runtime/article: "
        f"{total_runtime / successful_requests:.2f}s"
        if successful_requests
        else "Average runtime/article: N/A"
    )
    print()
    print(f"JSON results: {OUTPUT_FILE}")
    print(f"CSV results:  {CSV_FILE}")


if __name__ == "__main__":
    main()
