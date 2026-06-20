import torch


def generate_hf(model, tokenizer, input_ids, attention_mask, method, dataset=None):
    generate_kwargs = {}
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            generate_kwargs.update(
                {
                    "min_length": input_ids.shape[1] + 1,
                    "eos_token_id": [tokenizer.eos_token_id, newline_ids[-1]],
                }
            )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=method.max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
    return output_ids[:, input_ids.shape[1] :]
