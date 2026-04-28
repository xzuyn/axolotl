"""Module containing the CustomLLaMa3PromptTokenizingStrategy class"""

try:
    import ftfy
except ImportError:
    raise ImportError("You need ftfy. https://pypi.org/project/ftfy/")
import logging

# Import from axolotl package
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


class CustomLLaMa3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomLLaMa3.
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
        try:
            if self.tokenizer.bos_token_id is not None:
                all_input_ids, all_attention_mask, all_labels = (
                    [self.tokenizer.bos_token_id],
                    [1],
                    [IGNORE_TOKEN_ID]
                )
            else:
                all_input_ids, all_attention_mask, all_labels = [], [], []

            role_dict = {
                # ShareGPT
                "system": "system",
                "human": "user",
                "gpt": "assistant",
                # Extra
                "human-chat": "user",
                "gpt-chat": "assistant",
                # OpenAI/messages
                "user": "user",
                "assistant": "assistant",
            }

            turn_segments = []
            for i, turn in enumerate(prompt["prompt"]):
                if turn["from"] in ["human-chat", "gpt-chat"]:
                    sharegpt_value = ftfy.fix_text(
                        f"{turn['name'].strip()}: {turn['value'].strip()}"
                    )
                else:
                    sharegpt_value = ftfy.fix_text(turn["value"].strip())

                prefix_text = (
                    f"<|start_header_id|>{role_dict[turn['from']]}<|end_header_id|>\n\n"
                )

                tokenized_text = self.tokenizer(
                    text=f"{prefix_text}{sharegpt_value}<|eot_id|>",
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )

                # All prompt turns are masked
                turn_segments.append(
                    {
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": [IGNORE_TOKEN_ID] * len(tokenized_text["input_ids"]),
                    }
                )

            prefix_text = f"<|start_header_id|>assistant<|end_header_id|>\n\n"

            tokenized_text, regex_labels = regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=f"{prefix_text}{ftfy.fix_text(prompt['response'].strip())}<|eot_id|>",
            )

            prefix_token_count = 0
            for start, end in tokenized_text["offset_mapping"]:
                if end <= len(prefix_text):
                    prefix_token_count += 1
                else:
                    break

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
            for turn_segment in turn_segments:
                all_input_ids.extend(turn_segment["input_ids"])
                all_attention_mask.extend(turn_segment["attention_mask"])
                all_labels.extend(turn_segment["labels"])

            # Training on samples with all tokens masked is a waste of compute
            # May be worth checking if less than X% of tokens are trainable too
            if all(label == IGNORE_TOKEN_ID for label in all_labels):
                LOG.warning(
                    f"Processed sample will return empty due to no trainable tokens after masking"
                )
                return {"input_ids": [], "attention_mask": [], "labels": []}

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels,
            }
        except Exception as e:
            LOG.warning(e)
            return {"input_ids": [], "attention_mask": [], "labels": []}


# Function to load the CustomLLaMa3PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomLLaMa3PromptTokenizingStrategy(
        None, tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
