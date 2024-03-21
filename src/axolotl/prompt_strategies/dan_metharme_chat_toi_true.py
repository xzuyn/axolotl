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

        needs_bos = True
        chat_needs_model_tag = True
        # Iterate over each conversation part in the prompt
        for _, part in enumerate(self.prompter.build_prompt(prompt["conversations"])):
            if len(part) == 3:
                sharegpt_from, sharegpt_name, sharegpt_value = part
            else:
                sharegpt_from, sharegpt_value = part

            if sharegpt_from == "system":
                prefix = "<|system|>"
                if needs_bos is True:
                    res = self._tokenize(
                        prefix + sharegpt_value.strip(),
                        add_eos_token=False,
                        strip_bos_token=False,
                    )
                    labels = [*copy.deepcopy(res["input_ids"])]
                    needs_bos = False
                    chat_needs_model_tag = True
                else:
                    res = self._tokenize(
                        prefix + sharegpt_value.strip(),
                        add_eos_token=False,
                        strip_bos_token=True,
                    )
                    labels = [*copy.deepcopy(res["input_ids"])]
                    needs_bos = False
                    chat_needs_model_tag = True
            elif sharegpt_from == "human":
                prefix = "<|user|>"
                if needs_bos is True:
                    res = self._tokenize(
                        prefix + sharegpt_value.strip(),
                        add_eos_token=False,
                        strip_bos_token=False,
                    )
                    labels = [*copy.deepcopy(res["input_ids"])]
                    needs_bos = False
                    chat_needs_model_tag = True
                else:
                    res = self._tokenize(
                        prefix + sharegpt_value.strip(),
                        add_eos_token=False,
                        strip_bos_token=True,
                    )
                    labels = [*copy.deepcopy(res["input_ids"])]
                    needs_bos = False
                    chat_needs_model_tag = True
            elif sharegpt_from == "gpt":
                prefix = "<|model|>"
                if needs_bos is True:
                    res = self._tokenize(
                        prefix + sharegpt_value.strip(),
                        add_eos_token=True,
                        strip_bos_token=False,
                    )
                    labels = [*copy.deepcopy(res["input_ids"])]
                    needs_bos = False
                    chat_needs_model_tag = True
                else:
                    res = self._tokenize(
                        prefix + sharegpt_value.strip(),
                        add_eos_token=True,
                        strip_bos_token=True,
                    )
                    labels = [*copy.deepcopy(res["input_ids"])]
                    needs_bos = False
                    chat_needs_model_tag = True
            elif sharegpt_from == "human-chat":
                if chat_needs_model_tag is True:
                    if needs_bos is True:
                        prefix = f"<|model|>\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=False,
                            strip_bos_token=False,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
                    else:
                        prefix = f"<|model|>\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=False,
                            strip_bos_token=True,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
                else:
                    if needs_bos is True:
                        prefix = f"\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=False,
                            strip_bos_token=False,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
                    else:
                        prefix = f"\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=False,
                            strip_bos_token=True,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
            elif sharegpt_from == "gpt-chat":
                if chat_needs_model_tag is True:
                    if needs_bos is True:
                        prefix = f"<|model|>\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=True,
                            strip_bos_token=False,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
                    else:
                        prefix = f"<|model|>\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=True,
                            strip_bos_token=True,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
                else:
                    if needs_bos is True:
                        prefix = f"\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=True,
                            strip_bos_token=False,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
                    else:
                        prefix = f"\n{sharegpt_name}: "
                        res = self._tokenize(
                            prefix + sharegpt_value.strip(),
                            add_eos_token=True,
                            strip_bos_token=True,
                        )
                        labels = [*copy.deepcopy(res["input_ids"])]
                        needs_bos = False
                        chat_needs_model_tag = False
            else:
                # If the 'sharegpt_from' is unknown, issue a warning
                LOG.warning(f"unknown 'from' in conversation: {sharegpt_from}")
                res = defaultdict(lambda: [])

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
        PygmalionPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
