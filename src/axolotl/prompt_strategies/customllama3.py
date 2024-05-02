"""Module containing the CustomLLaMa3PromptTokenizingStrategy and CustomLLaMa3Prompter class"""

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


class CustomLLaMa3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomLLaMa3.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)

    def tokenize_prompt(self, prompt):
        # Tokenize the prompt based on its conversations
        result, current_len = tokenize_prompt_default()

        strip_bos = False

        # Iterate over each conversation part in the prompt
        num_parts = len(prompt["conversations"])
        for i, part in enumerate(self.prompter.build_prompt(prompt["conversations"])):
            if i == num_parts - 1:
                end_of_text = True
            else:
                end_of_text = False

            if len(part) == 3:
                sharegpt_from, sharegpt_name, sharegpt_value = part

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
            elif len(part) == 2:
                sharegpt_from, sharegpt_value = part

                if sharegpt_from == "system":
                    role_name = "system"
                elif sharegpt_from == "human":
                    role_name = "user"
                elif sharegpt_from == "gpt":
                    role_name = "assistant"
            else:
                LOG.warning(f"unknown 'len(part)' in conversation: {len(part)}")
                exit()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self._tokenize(
                f"<|start_header_id|>{role_name}<|end_header_id|>\n\n",
                add_eos_token=False,
                strip_bos_token=strip_bos,
            )

            res = self._tokenize(
                f"<|start_header_id|>{role_name}<|end_header_id|>\n\n{sharegpt_value.strip()}<|eot_id|>",
                add_eos_token=end_of_text,
                strip_bos_token=strip_bos,
            )

            if (
                self.train_on_inputs is False
                and (sharegpt_from == "system" or sharegpt_from == "human" or sharegpt_from == "human-chat")
            ):
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            elif (
                self.train_on_inputs is False
                and (sharegpt_from == "gpt" or sharegpt_from == "gpt-chat")
            ):
                labels = (
                    [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
                    + [*copy.deepcopy(res["input_ids"])][len(prefix["input_ids"]):]
                )
            else:
                labels = res["input_ids"]

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


class CustomLLaMa3Prompter:
    """
    Prompter for CustomLLaMa3.
    """

    def __init__(self, *args, **kwargs):
        # Constructor does nothing
        pass

    def build_prompt(
        self, source, *args, **kwargs  # pylint: disable=unused-argument
    ) -> Generator[Tuple[str, str], None, None]:
        # Generator function to yield 'from' and 'value' or 'from', 'name', and 'value' tuples
        for msg in source:
            if "name" in msg:
                yield msg["from"], msg["name"], msg["value"]
            else:
                yield msg["from"], msg["value"]


def load(tokenizer, cfg):
    # Function to load the CustomLLaMa3PromptTokenizingStrategy
    return CustomLLaMa3PromptTokenizingStrategy(
        CustomLLaMa3Prompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )
