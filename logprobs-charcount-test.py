"""Set environment variables for offline mode"""
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from mlx_lm import load
import mlx.core as mx
import cmudict
import time

# Initialize CMU dictionary for syllable counting
cmu_dict = cmudict.dict()
# print(f"{cmu_dict['logic']}")

word = "pixelated"
print(f"{word} - {sum(1 for char in word if char in 'aeiouAEIOU')}")


def count_syllables(word):
    """Count syllables in a word using CMU dictionary."""
    word_lower = word.lower()
    if word_lower in cmu_dict:
        # CMU dict returns list of pronunciations, we use the first one
        # Each digit in pronunciation represents a stressed vowel (syllable)
        return len([ph for ph in cmu_dict[word_lower][0] if ph[-1].isdigit()])
    # Fallback: count vowels as rough approximation
    vowels = "aeiouAEIOU"
    return sum(1 for char in word if char in vowels)


def select_token(tokenizer, current_buffer, sorted_tokens):
    if not len(sorted_tokens):
        return None

    str = tokenizer.decode([int(sorted_tokens[0])])
    print(f"{time.time()}: {str!r}")
    stripped = str.strip()

    # select our prompted eol token or any whitespace tokens
    if stripped == "/" or len(stripped) == 0:
        return sorted_tokens[0]

    # skip any token that is a single non-alphabetic character
    if len(stripped) == 1 and not stripped.isalpha():
        return select_token(tokenizer, current_buffer, sorted_tokens[1:])

    if count_syllables(stripped) == 2:
        return sorted_tokens[0]

    return select_token(tokenizer, current_buffer, sorted_tokens[1:])


def main():
    model, tokenizer = load("mlx-community/gemma-3-4b-it-8bit")
    # model, tokenizer = load("mlx-community/gemma-3-12b-it-4bit")
    # model, tokenizer = load("mlx-community/gemma-3-12b-it-8bit")
    # model, tokenizer = load("mlx-community/gemma-3-27b-it-8bit")

    def gen_single(prompt):
        # messages = [{"role": "user", "content": prompt}]
        # prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # print(f"Prompt:\n{tokenizer.decode(prompt)!r}")

        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Convert prompt to MLX array if it isn't already
        prompt = mx.array(prompt)

        # Encode the suffix and convert to MLX array
        # suffix = mx.array(tokenizer.encode("Functions", add_special_tokens=False))
        # for token in suffix:
        #     print(f"{token}: {tokenizer.decode([int(token)])}")
        # prompt = mx.concatenate([prompt, suffix])

        print(f"Prompt:\n{tokenizer.decode(prompt.tolist())!r}")

        output_tokens = []  # all tokens we accept and accumulate for final output
        current_buffer = []  # the in‐flight word
        current_word = ""

        # Initialize context with the prompt
        context = mx.array(prompt)

        # Generate tokens one at a time
        for _ in range(4096):
            # generate the next token
            logits = model(context[None])

            # Get log probabilities
            log_probs = logits[0, -1, :] - mx.logsumexp(logits[0, -1, :])
            tops = mx.argsort(-log_probs)[:40]

            # we're done if end of turn
            if tops[0] == tokenizer.eos_token_id:
                break

            for i, tid in enumerate(tops):
                lp = float(log_probs[tid])
                token_text = tokenizer.decode([int(tid)])

            # select the next token
            next_token = select_token(tokenizer, current_buffer, tops)

            selected = next_token
            output_tokens.append(selected)
            token_text = tokenizer.decode([int(selected)])
            # print(f"---- selected: {token_text!r}")

            # update context for next iteration (pre-fill)
            context = mx.concatenate([context, mx.array([int(selected)])])

        final = tokenizer.decode(output_tokens)
        print("Final output:")
        print(final)

    gen_single(
        "Write a short poem about computer programming. The poem should have 4 lines. At the end of each line, add this punctuation ' /'\n\nOutput only the poem, nothing else."
    )


if __name__ == "__main__":
    main()
