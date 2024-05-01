"""Module containing the PygmalionPromptTokenizingStrategy and PygmalionPrompter class"""

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


class PygmalionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for Pygmalion.
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
                    role_name = "role: system"
                elif sharegpt_from == "human":
                    role_name = "role: user"
                elif sharegpt_from == "human-chat":
                    role_name = f"role: user | name: {sharegpt_name}"
                elif sharegpt_from == "gpt":
                    role_name = "role: assistant"
                elif sharegpt_from == "gpt-chat":
                    role_name = f"role: assistant | name: {sharegpt_name}"
            elif len(part) == 2:
                sharegpt_from, sharegpt_value = part

                if sharegpt_from == "system":
                    role_name = "role: system"
                elif sharegpt_from == "human":
                    role_name = "role: user"
                elif sharegpt_from == "gpt":
                    role_name = "role: assistant"
            else:
                LOG.warning(f"unknown 'len(part)' in conversation: {len(part)}")
                exit()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self._tokenize(
                (
                    "<|im_start|>"
                    + role_name
                    + "\n"
                ),
                add_eos_token=False,
                strip_bos_token=strip_bos,
            )
            message = self._tokenize(
                sharegpt_value.strip(),
                add_eos_token=False,
                strip_bos_token=True,
            )
            suffix = self._tokenize(
                "<|im_end|>",
                add_eos_token=end_of_text,
                strip_bos_token=True,
            )

            res = self._tokenize(
                f"<|im_start|>{role_name}\n{sharegpt_value.strip()}<|im_end|>",
                add_eos_token=end_of_text,
                strip_bos_token=strip_bos,
            )

            if (
                self.train_on_inputs is False
                and (sharegpt_from == "system" or sharegpt_from == "human" or sharegpt_from == "human-chat")
            ):
                labels = (
                    [*copy.deepcopy(prefix["input_ids"])]  # Keep the prefix
                    + [IGNORE_TOKEN_ID] * len([*copy.deepcopy(message["input_ids"])])  # Mask out the user message
                    + [*copy.deepcopy(suffix["input_ids"])]  # Keep the suffix
                )
            else:
                labels = [*copy.deepcopy(res["input_ids"])]

            # Make sure the BOS token is unmasked
            if strip_bos is False:
                del labels[0]
                labels.insert(0, self.tokenizer.bos_token_id)

            # Make sure the EOS token is unmasked
            if end_of_text:
                del labels[-1]
                labels.append(self.tokenizer.eos_token_id)

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


class PygmalionPrompter:
    """
    Prompter for Pygmalion.
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
    # Function to load the PygmalionPromptTokenizingStrategy
    return PygmalionPromptTokenizingStrategy(
        PygmalionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )
