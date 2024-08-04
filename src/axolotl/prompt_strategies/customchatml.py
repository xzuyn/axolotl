"""Module containing the CustomChatMLPromptTokenizingStrategy class"""

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


class CustomChatMLPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomChatML.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)

    def tokenize_prompt(self, prompt):
        # Tokenize the prompt based on its conversations
        result, current_len = tokenize_prompt_default()

        # We don't want to remove the BOS token for the first turn
        strip_bos = False

        # Sometimes it gets named 'conversations' and other times 'conversation'
        if "conversations" in prompt:
            conversation_name = "conversations"
        elif "conversation" in prompt:
            conversation_name = "conversation"
        else:
            LOG.warning(f"sample does not contain 'conversations' or 'conversation'")
            exit()

        num_turns = len(prompt[conversation_name])

        # Iterate over each conversation turn in the prompt
        for i, turn in enumerate(prompt[conversation_name]):
            # Check if this is the last turn, so we know to add the EOS token
            if i == num_turns - 1:
                end_of_text = True
            else:
                end_of_text = False

            # Add a new line to the beginning if it's not the first turn
            if i == 0:
                add_new_line = ""
            else:
                add_new_line = "\n"

            sharegpt_from, sharegpt_value = turn["from"], turn["value"]
            if sharegpt_from == "system":
                role_name = "system"
            elif sharegpt_from == "human":
                role_name = "user"
            elif sharegpt_from == "human-chat":
                role_name = "user"
                sharegpt_value = f"{turn['name']}: {sharegpt_value}"
            elif sharegpt_from == "gpt":
                role_name = "model"
            elif sharegpt_from == "gpt-chat":
                role_name = "model"
                sharegpt_value = f"{turn['name']}: {sharegpt_value}"
            else:
                LOG.warning(f"'from' contains an unhandled string")
                exit()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self._tokenize(
                f"{add_new_line}<|im_start|>{role_name}\n",
                add_eos_token=False,
                strip_bos_token=strip_bos,
            )

            # Get entire tokenized turn
            res = self._tokenize(
                f"{add_new_line}<|im_start|>{role_name}\n"
                f"{sharegpt_value.strip()}<|im_end|>",
                add_eos_token=end_of_text,
                strip_bos_token=strip_bos,
            )

            # Handle masked user turn
            if (
                self.train_on_inputs is False
                and (
                    sharegpt_from == "system"
                    or sharegpt_from == "human"
                    or sharegpt_from == "human-chat"
                )
            ):
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            # Handle partially masked model turn
            elif (
                self.train_on_inputs is False
                and (
                    sharegpt_from == "gpt"
                    or sharegpt_from == "gpt-chat"
                )
            ):
                labels = (
                    [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
                    + [*copy.deepcopy(res["input_ids"])][len(prefix["input_ids"]):]
                )
            # Handle unmasked turn
            else:
                labels = res["input_ids"]

            # Now that we've done the first turn we can remove the BOS token
            strip_bos = True

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
class CustomChatMLPrompter:
    """
    Prompter for CustomChatML.
    """

    def __init__(self, *args, **kwargs):
        # Constructor does nothing
        pass


# Function to load the CustomChatMLPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomChatMLPromptTokenizingStrategy(
        CustomChatMLPrompter(),  # TODO: Remove this as it doesn't get used
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )
