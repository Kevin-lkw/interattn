import argparse

from ..common import add_generation_args, run_generation_benchmark


def ruler_prompt(context, question, record):
    instruction = record.get("instruction")
    if instruction:
        return str(instruction)
    if question:
        return f"{context}\n\nQuestion: {question}\nAnswer:"
    return context


def parse_args():
    parser = add_generation_args(
        argparse.ArgumentParser(description="Run RULER-style generation.")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_generation_benchmark(
        args,
        benchmark_name="ruler",
        prompt_builder=ruler_prompt,
    )


if __name__ == "__main__":
    main()
