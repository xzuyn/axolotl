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

            # Check if the conversation is CustomShareGPT
            if "from" in turn and "name" in turn and "value" in turn:
                sharegpt_from, sharegpt_name, sharegpt_value = turn["from"], turn["name"], turn["value"]

                if sharegpt_from == "system":
                    role_name = "system"
                elif sharegpt_from == "human":
                    role_name = "user"
                elif sharegpt_from == "gpt":
                    role_name = "assistant"
                elif sharegpt_from == "human-chat":
                    role_name = f"{sharegpt_name}"
                elif sharegpt_from == "gpt-chat":
                    role_name = f"{sharegpt_name}"
                elif sharegpt_from == "human-tool":
                    role_name = f"tool request: {sharegpt_name}"
                elif sharegpt_from == "gpt-tool":
                    role_name = f"tool response: {sharegpt_name}"
                else:
                    LOG.warning(f"'from' contains an unhandled string")
                    exit()
            # Check if the conversation is ShareGPT
            elif "from" in turn and "value" in turn:
                sharegpt_from, sharegpt_value = turn["from"], turn["value"]

                if sharegpt_from == "system":
                    role_name = "system"
                elif sharegpt_from == "human":
                    role_name = "user"
                elif sharegpt_from == "gpt":
                    role_name = "assistant"
                else:
                    LOG.warning(f"'from' contains an unhandled string")
                    exit()
            else:
                LOG.warning(f"conversation does not contain 'from' or 'value'")
                exit()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self._tokenize(
                f"<|im_start|>{role_name}\n",
                add_eos_token=False,
                strip_bos_token=strip_bos,
            )

            # Get entire tokenized turn
            res = self._tokenize(
                f"<|im_start|>{role_name}\n{sharegpt_value.strip()}<|im_end|>",
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
                    or sharegpt_from == "human-tool"
                )
            ):
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            # Handle partially masked model turn
            elif (
                self.train_on_inputs is False
                and (
                    sharegpt_from == "gpt"
                    or sharegpt_from == "gpt-chat"
                    or sharegpt_from == "gpt-tool"
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