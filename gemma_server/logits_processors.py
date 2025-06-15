import mlx.core as mx
import cmudict

__all__ = ["top_k_logits_processor", "SyllableLogitsProcessor"]


def top_k_logits_processor(tokens: mx.array, logits: mx.array, k: int = 50) -> mx.array:
    """Top-k logits processor that works properly with MLX."""
    kth_largest = mx.sort(logits, axis=-1)[:, -k][:, None]
    mask = logits >= kth_largest
    return mx.where(mask, logits, -mx.inf)


class SyllableLogitsProcessor:
    """Filter logits so only tokens with the target syllable count are allowed."""

    def __init__(self, tokenizer, cmu_dict, target_syllable_count: int):
        self.tokenizer = tokenizer
        self.cmu_dict = cmu_dict
        self.target_syllable_count = target_syllable_count
        self.valid_token_mask = None

    def count_syllables(self, word: str) -> int:
        word_lower = word.lower()
        if word_lower in self.cmu_dict:
            return len([ph for ph in self.cmu_dict[word_lower][0] if ph[-1].isdigit()])
        vowels = "aeiouAEIOU"
        return sum(1 for char in word if char in vowels)

    def _build_mask(self, vocab_size: int) -> mx.array:
        mask = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        max_token = min(vocab_size, tokenizer_vocab_size)

        for token_id in range(vocab_size):
            if token_id >= max_token:
                mask.append(False)
                continue
            try:
                if eos_token_id is not None and token_id == eos_token_id:
                    mask.append(True)
                    continue

                token_text = self.tokenizer.decode([token_id]).strip()

                if token_text in [".", ",", "!", "?", "/"] or len(token_text) == 0:
                    mask.append(True)
                    continue

                if len(token_text) == 1 and not token_text.isalpha():
                    mask.append(False)
                    continue

                if self.count_syllables(token_text) == self.target_syllable_count:
                    mask.append(True)
                else:
                    mask.append(False)
            except Exception:
                mask.append(False)

        return mx.array(mask)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.valid_token_mask is None or self.valid_token_mask.shape[0] != logits.shape[-1]:
            self.valid_token_mask = self._build_mask(logits.shape[-1])

        filtered_logits = mx.where(self.valid_token_mask[None, :], logits, -mx.inf)

        if mx.all(mx.isinf(filtered_logits)):
            return logits

        return filtered_logits
