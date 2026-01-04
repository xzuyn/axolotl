"""Module containing the CustomGemma3PromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
import logging
import random
from axolotl.prompt_tokenizers import PromptTokenizingStrategy

try:
    from axolotl.prompt_strategies.regex_attention import regex_attention_tokenizer
except ImportError:
    raise ImportError(
        "You need https://github.com/xzuyn/axolotl/blob/latest-formatters/src/axolotl/prompt_strategies/regex_attention.py"
    )


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


class CustomGemma3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomGemma3.
    """

    def __init__(
        self, prompter, tokenizer, train_on_inputs, sequence_len, *args, **kwargs
    ):
        # Call the superclass' constructor
        super().__init__(
            prompter=prompter,
            tokenizer=tokenizer,
            train_on_inputs=train_on_inputs,
            sequence_len=sequence_len,
            *args,
            **kwargs,
        )

    def tokenize_prompt(self, prompt):
        if len(prompt["prompt"]) <= 2 or (1024 < prompt["response_average_tokens"] < 3):
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # Some tokenizers don't contain this, so if it doesn't exist assume it is set to True
        add_bos = getattr(self.tokenizer, "add_bos_token", True)

        # ShareGPT-to-Gemma3 Dictionary
        role_dict = {
            "system": "system",
            "user": "user",
            "human": "user",
            "human-chat": "user",
            "assistant": "model",
            "gpt": "model",
            "gpt-chat": "model",
        }

        best_prefill = None
        best_response = None
        best_slop_ratio = 1.0
        for response in prompt["responses"]:
            model = response.get("model")
            if model is None or "claude" not in model:
                continue

            slop_ratio = response.get("slop_ratio")
            response_tokens = response.get("response_tokens")
            minos_prediction = response.get("minos_prediction")
            minos_confidence = response.get("minos_confidence")
            if slop_ratio is None or response_tokens is None or minos_prediction is None or minos_confidence is None:
                continue

            if (
                slop_ratio < best_slop_ratio
                and 1024 >= response_tokens >= 3
                and minos_prediction == "non-refusal"
                and minos_confidence > 0.8
            ):
                best_prefill = response.get("prefill")
                best_response = response.get("response")
                best_slop_ratio = slop_ratio

        if best_response is None or best_response == "" or best_slop_ratio == 1.0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        prompt["prompt"].append(
            {"from": "gpt", "prefill": best_prefill, "value": best_response}
        )

        turn_segments = []
        token_count = 0
        for i, turn in enumerate(prompt["prompt"]):
            # Skip empty turns
            sharegpt_value = turn.get("value")
            if sharegpt_value is None or sharegpt_value.strip() == "":
                continue

            prefix_text = (
                "\n" if i != 0 else ""
            ) + f"<start_of_turn>{role_dict[turn['from']]}\n"

            # All turns except the final turn
            if i != len(prompt["prompt"]) - 1:
                # Use non-regex tokenizer cause its faster. Only final turn needs regex
                tokenized_text = self.tokenizer(
                    text=f"{prefix_text}{sharegpt_value}<end_of_turn>",
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                    return_offsets_mapping=True,
                )

                # Skip processing early if over token limit
                token_count += len(tokenized_text["input_ids"])
                if token_count >= self.sequence_len:
                    return {"input_ids": [], "attention_mask": [], "labels": []}

                # Add masked turn to turn segments
                turn_segments.append(
                    {
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": [IGNORE_TOKEN_ID] * len(tokenized_text["input_ids"]),
                    }
                )
            # Final turn
            else:
                # Add prefill to prefix_text if it exists
                prefill_text = turn.get("prefill")
                if prefill_text is not None and prefill_text.strip() != "":
                    prefix_token_count = 1  # Sometimes it seems to miss by 1 with prefill, so mask 1 extra
                    prefix_text += prefill_text
                else:
                    prefix_token_count = 0

                # Tokenize and create mask out undesired tokens using regex patterns
                tokenized_text, regex_labels = regex_attention_tokenizer(
                    tokenizer=self.tokenizer,
                    text=f"{prefix_text}{sharegpt_value}<end_of_turn>",
                )

                # Skip processing early if over token limit
                token_count += len(tokenized_text["input_ids"])
                if token_count >= self.sequence_len:
                    return {"input_ids": [], "attention_mask": [], "labels": []}

                for start, end in tokenized_text["offset_mapping"]:
                    if end <= len(prefix_text):
                        prefix_token_count += 1
                    else:
                        break

                # Add partially masked turn to turn segments
                turn_segments.append(
                    {
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": (
                            [IGNORE_TOKEN_ID] * prefix_token_count  # Mask the prefix
                            + regex_labels[prefix_token_count:]
                        ),
                    }
                )

        # Combine all the turn segments
        input_ids, attention_mask, labels = [], [], []
        for turn_segment in turn_segments:
            input_ids.extend(turn_segment["input_ids"])
            attention_mask.extend(turn_segment["attention_mask"])
            labels.extend(turn_segment["labels"])

        # Add missing BOS token if needed
        if (
            add_bos
            and self.tokenizer.bos_token_id
            and input_ids[0] != self.tokenizer.bos_token_id
        ):
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 1)
            labels.insert(0, IGNORE_TOKEN_ID)

        # Training on samples with all tokens masked is a waste of compute
        # May be worth checking if less than X% of tokens are trainable too
        if all(label == IGNORE_TOKEN_ID for label in labels):
            LOG.warning(
                f"Processed sample will return empty due to no trainable tokens after masking"
            )
            return {"input_ids": [], "attention_mask": [], "labels": []}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# Function to load the CustomGemma3PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomGemma3PromptTokenizingStrategy(
        None, tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
