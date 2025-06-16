"""Module containing the ePubCleanerPromptTokenizingStrategy class"""

# Import necessary modules and functions
import logging
from copy import deepcopy

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy

# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


class ePubCleanerPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for ePubCleaner.
    """

    def __init__(self, prompter, tokenizer, train_on_inputs, sequence_len, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(
            prompter=prompter,
            tokenizer=tokenizer,
            train_on_inputs=train_on_inputs,
            sequence_len=sequence_len,
            *args,
            **kwargs
        )

    def tokenize_prompt(self, prompt):
        # Some tokenizers don't contain this, so if it doesn't exist assume it is set to True
        add_bos = getattr(self.tokenizer, "add_bos_token", True)

        maybe_unclean_text = prompt["conversations"][0]["value"]  # human
        is_text_unclean = "<|text_is_unclean|>" if prompt["is_edited"] else "<|text_is_clean|>"

        prefix_text = f"<|text_check_start|>{maybe_unclean_text}<|text_check_end|>"
        tokenized_text = self.tokenizer(
            text=f"{prefix_text}{is_text_unclean}{self.tokenizer.eos_token}",
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_offsets_mapping=True,
        )

        if self.train_on_inputs is False:
            prefix_token_count = 0
            for start, end in tokenized_text["offset_mapping"]:
                if end <= len(prefix_text):
                    prefix_token_count += 1
                else:
                    break
            input_ids = tokenized_text["input_ids"]
            attention_mask = tokenized_text["attention_mask"]
            labels = (
                [IGNORE_TOKEN_ID] * prefix_token_count  # Mask the prefix
                + input_ids[prefix_token_count:]
            )
        else:
            input_ids = tokenized_text["input_ids"]
            attention_mask = tokenized_text["attention_mask"]
            labels = input_ids

        # Add missing BOS token if needed
        if add_bos and self.tokenizer.bos_token_id and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 1)
            labels.insert(0, IGNORE_TOKEN_ID)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# Function to load the ePubCleanerPromptTokenizingStrategy
def load(tokenizer, cfg):
    return ePubCleanerPromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs, cfg.sequence_len)
