"""Module containing the CustomCompletionPromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
try:
    import ftfy
except ImportError:
    raise ImportError("You need ftfy. https://pypi.org/project/ftfy/")
import logging
from copy import deepcopy

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy
try:
    from axolotl.prompt_strategies.formatter_regex import COMPILED_REGEX_PATTERNS
except ImportError:
    raise ImportError("You need https://github.com/xzuyn/axolotl/blob/came-plus-formatters/src/axolotl/prompt_strategies/formatter_regex.py")


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


def mask_regex_attention_tokenizer(tokenizer, text, compiled_regex_patterns, add_special_tokens=False):
    tokenized_text = tokenizer(
        text=text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
        return_tensors=None,
        return_offsets_mapping=True,
    )

    regex_mask_labels = deepcopy(tokenized_text["input_ids"])
    for pattern in compiled_regex_patterns:
        for match in pattern.finditer(text):
            found_index = match.start()
            end_index = match.end()

            # Check each token's character span; if it overlaps, mask it out.
            for i, (token_start, token_end) in enumerate(tokenized_text["offset_mapping"]):
                if token_start < end_index and token_end > found_index:
                    regex_mask_labels[i] = IGNORE_TOKEN_ID

    return tokenized_text, regex_mask_labels


class CustomCompletionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomCompletion.
    """

    def __init__(self, prompter, tokenizer, field, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)
        self.field = "text" if not field else field

    def tokenize_prompt(self, prompt):
        # Tokenize and create mask out undesired tokens using regex patterns
        tokenized_text, regex_mask_labels = mask_regex_attention_tokenizer(
            tokenizer=self.tokenizer,
            text=ftfy.fix_text(prompt[self.field]).strip(),
            compiled_regex_patterns=COMPILED_REGEX_PATTERNS,
        )

        # Add missing BOS token
        if self.tokenizer.bos_token_id and tokenized_text["input_ids"][0] != self.tokenizer.bos_token_id:
            tokenized_text["input_ids"].insert(0, self.tokenizer.bos_token_id)
            tokenized_text["attention_mask"].insert(0, 1)
            regex_mask_labels.insert(0, IGNORE_TOKEN_ID)

        # Add missing EOS token
        if tokenized_text["input_ids"][-1] != self.tokenizer.eos_token_id:
            tokenized_text["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_text["attention_mask"].append(1)
            regex_mask_labels.append(self.tokenizer.eos_token_id)

        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "labels": regex_mask_labels
        }


# Function to load the CustomCompletionPromptTokenizingStrategy
def load(tokenizer, cfg, ds_cfg):
    return CustomCompletionPromptTokenizingStrategy(None, tokenizer, ds_cfg.field)
