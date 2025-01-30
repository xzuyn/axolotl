"""Module containing the CustomMistralV7PromptTokenizingStrategy class"""

# Import necessary modules and functions
import copy
import logging
from collections import defaultdict
from typing import Generator, List, Tuple

# Import from axolotl package
from axolotl.prompt_tokenizers import (
    PromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)

# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


class CustomMistralV7PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomMistralV7.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)

    def tokenize_prompt(self, prompt):
        # Tokenize the prompt based on its conversations
        result, current_len = tokenize_prompt_default()

        # Sometimes it gets named 'conversations' and other times 'conversation'
        if "conversations" in prompt:
            conversation_name = "conversations"
        elif "conversation" in prompt:
            conversation_name = "conversation"
        else:
            LOG.warning(f"sample does not contain 'conversations' or 'conversation'")
            exit()

        # Iterate over each conversation turn in the prompt
        num_turns = len(prompt[conversation_name])
        for i, turn in enumerate(prompt[conversation_name]):
            # Strip BOS token if it's not the first turn
            if i == 0:
                strip_bos = False
            else:
                strip_bos = True

            # Check if this is the last turn, so we know to add the EOS token
            if i == num_turns - 1:
                end_of_text = True
            else:
                end_of_text = False

            # Get correct roles and messages
            sharegpt_from, sharegpt_value = turn["from"].strip(), turn["value"].strip()

            # ShareGPT Roles
            if sharegpt_from == "system":
                prepped_content = f"[SYSTEM_PROMPT]{sharegpt_value}[/SYSTEM_PROMPT]"
            elif sharegpt_from == "human":
                prepped_content = f"[INST]{sharegpt_value}[/INST]"
            elif sharegpt_from == "gpt":
                prepped_content = f"{sharegpt_value}"
                end_of_text = True
            # CustomShareGPT Roles
            elif sharegpt_from == "human-chat":
                prepped_content = f"[INST]{turn['name'].strip()}: {sharegpt_value}[/INST]"
            elif sharegpt_from == "gpt-chat":
                prepped_content = f"{turn['name'].strip()}: {sharegpt_value}"
                end_of_text = True
            else:
                LOG.warning(f"'from' contains an unhandled string: {sharegpt_from}")
                exit()

            # Get entire tokenized turn
            res = self.tokenizer(
                prepped_content,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            if res["input_ids"][-1] != self.tokenizer.eos_token_id and end_of_text:
                res["input_ids"].append(self.tokenizer.eos_token_id)
                res["attention_mask"].append(1)
            if res["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos:
                res["input_ids"] = res["input_ids"][1:]
                res["attention_mask"] = res["attention_mask"][1:]

            # Handle masked user turn
            if self.train_on_inputs is False and (
                sharegpt_from == "system"
                or sharegpt_from == "human"
                or sharegpt_from == "human-chat"
            ):
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            # Handle unmasked turn
            else:
                labels = res["input_ids"]

            # Parse tokenized result and update current length
            result, current_len = parse_tokenized_to_result(
                result,
                current_len,
                res,
                labels,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return result


# TODO: Remove this as it doesn't get used
class CustomMistralV7Prompter:
    """
    Prompter for CustomMistralV7.
    """

    def __init__(self, *args, **kwargs):
        # Constructor does nothing
        pass


# Function to load the CustomMistralV7PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomMistralV7PromptTokenizingStrategy(
        CustomMistralV7Prompter(),  # TODO: Remove this as it doesn't get used
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )

