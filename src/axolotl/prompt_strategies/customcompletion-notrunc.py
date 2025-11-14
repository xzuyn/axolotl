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
            **kwargs
        )
        self.field = "text" if not field else field

    def tokenize_prompt(self, prompt):
        # Some tokenizers don't contain this, so if it doesn't exist assume it is set to True
        add_bos = getattr(self.tokenizer, "add_bos_token", True)

        try:
            tokenized_text = self.tokenizer(
                text=ftfy.fix_text(prompt[self.field]).strip(),
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            labels = deepcopy(tokenized_text["input_ids"])
        except AttributeError:
            LOG.warning(f"Processed sample will return empty due to AttributeError")
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }

        try:
            # Add missing BOS token
            if add_bos and self.tokenizer.bos_token_id and tokenized_text["input_ids"][0] != self.tokenizer.bos_token_id:
                tokenized_text["input_ids"].insert(0, self.tokenizer.bos_token_id)
                tokenized_text["attention_mask"].insert(0, 1)
                labels.insert(0, IGNORE_TOKEN_ID)

            # Add missing EOS token
            if tokenized_text["input_ids"][-1] != self.tokenizer.eos_token_id:
                tokenized_text["input_ids"].append(self.tokenizer.eos_token_id)
                tokenized_text["attention_mask"].append(1)
                labels.append(self.tokenizer.eos_token_id)
        except IndexError:
            LOG.warning(f"Processed sample will return empty due to IndexError")
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }

        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "labels": labels
        }


# Function to load the CustomCompletionPromptTokenizingStrategy
def load(tokenizer, cfg, ds_cfg):
    return CustomCompletionPromptTokenizingStrategy(None, tokenizer, ds_cfg.field, cfg.sequence_len)
