from QEfficient.generation.spec_prefill import SpecPrefillEngine
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

    engine = SpecPrefillEngine(
        tokenizer=tokenizer,
        qpc_path="compiled_spec.qpc",
        ctx_len=2048,
        device_id=[0],
    )

    outputs, position_ids, generation_len = engine.run_prefill(["I am"], generation_len=1)
    # SpecPrefillEngine will print: "Prefill last token: <decoded_token>"


if __name__ == "__main__":
    main()
