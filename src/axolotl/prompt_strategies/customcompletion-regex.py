"""Module containing the CustomCompletionPromptTokenizingStrategy class"""

# Import necessary modules and functions
import re

try:
    import ftfy
except ImportError:
    raise ImportError("You need ftfy. https://pypi.org/project/ftfy/")
import logging

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy

try:
    from axolotl.prompt_strategies.regex_attention import (
        COMPILED_REGEX_PATTERNS,
        regex_attention_tokenizer,
    )
except ImportError:
    raise ImportError(
        "You need https://github.com/xzuyn/axolotl/blob/latest-formatters/src/axolotl/prompt_strategies/regex_attention.py"
    )


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


class CustomCompletionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomCompletion.
    """

    def __init__(self, prompter, tokenizer, field, sequence_len, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(
            prompter=prompter,
            tokenizer=tokenizer,
            sequence_len=sequence_len,
            *args,
            **kwargs,
        )
        self.field = "text" if not field else field

    def tokenize_prompt(self, prompt):
        # Some tokenizers don't contain this, so if it doesn't exist assume it is set to True
        add_bos = getattr(self.tokenizer, "add_bos_token", True)

        # Tokenize and create mask out undesired tokens using regex patterns
        try:
            tokenized_text, regex_labels = regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=ftfy.fix_text(prompt[self.field]).strip(),
                compiled_regex_patterns=COMPILED_REGEX_PATTERNS,
            )
        except AttributeError:
            LOG.warning(f"Processed sample will return empty due to AttributeError")
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # Add missing BOS token
        if (
            add_bos
            and self.tokenizer.bos_token_id
            and tokenized_text["input_ids"][0] != self.tokenizer.bos_token_id
        ):
            tokenized_text["input_ids"].insert(0, self.tokenizer.bos_token_id)
            tokenized_text["attention_mask"].insert(0, 1)
            regex_labels.insert(0, IGNORE_TOKEN_ID)

        # Add missing EOS token
        if tokenized_text["input_ids"][-1] != self.tokenizer.eos_token_id:
            tokenized_text["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_text["attention_mask"].append(1)
            regex_labels.append(self.tokenizer.eos_token_id)

        # Training on samples with all tokens masked is a waste of compute
        # May be worth checking if less than X% of tokens are trainable too
        if all(label == IGNORE_TOKEN_ID for label in regex_labels):
            LOG.warning(
                f"Processed sample will return empty due to no trainable tokens after masking"
            )
            return {"input_ids": [], "attention_mask": [], "labels": []}

        if len(tokenized_text["input_ids"]) <= self.sequence_len:
            return {
                "input_ids": tokenized_text["input_ids"],
                "attention_mask": tokenized_text["attention_mask"],
                "labels": regex_labels,
            }
        else:
            return {
                "input_ids": tokenized_text["input_ids"][: self.sequence_len],
                "attention_mask": tokenized_text["attention_mask"][: self.sequence_len],
                "labels": regex_labels[: self.sequence_len],
            }


# Function to load the CustomCompletionPromptTokenizingStrategy
def load(tokenizer, cfg, ds_cfg):
    return CustomCompletionPromptTokenizingStrategy(
        None, tokenizer, ds_cfg.field, cfg.sequence_len
    )
