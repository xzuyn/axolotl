"""Module containing the CustomLLaMa3TrashLogsV3PromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
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
URL_FINDING_REGEX_PATTERN = (
    r"\b(?:https?|ftp|smtp):\/\/"  # Word boundary + protocol
    r"([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}"  # Domain name
    r"(:\d{1,5})?"  # Optional port
    r"(\/[a-zA-Z0-9#\/?=&._-]*)?"  # Path, query parameters, or fragments
    r"\b"  # Word boundary to prevent partial matches
)


class CustomLLaMa3TrashLogsV3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomLLaMa3TrashLogsV3.
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

            # Add attachment info if it exists
            if turn["attachments"]:
                turn_value = (
                    f"[Attachments: {', '.join(attachment['name'] for attachment in turn['attachments'])}]\n\n"
                    f"{turn['value']}"
                ).strip()
            elif turn["stickers"]:
                turn_value = (
                    f"[Stickers: {', '.join(sticker['name'] for sticker in turn['stickers'])}]\n\n"
                    f"{turn['value']}"
                ).strip()
            else:
                turn_value = turn["value"].strip()

            turn_from = f"messageid: {turn['messageid']} | timestamp: {turn['timestamp']} | username: {turn['name']} | nickname: {turn['nickname']} | type: {turn['type']}"
            if turn["mentions"]:
                turn_from += f" | mentions: {', '.join(mention['name'] for mention in turn['mentions'])}"
            if turn["type"] == "Reply":
                turn_from += f" | reference: {turn['reference']}"

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self.tokenizer(
                (
                    f"<|start_header_id|>{turn_from}<|end_header_id|>\n\n"
                ),
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            if prefix["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos:
                prefix["input_ids"] = prefix["input_ids"][1:]
                prefix["attention_mask"] = prefix["attention_mask"][1:]

            # Get entire tokenized turn
            res = self.tokenizer(
                (
                    f"<|start_header_id|>{turn_from}<|end_header_id|>\n\n"
                    f"{turn_value}<|eot_id|>"
                ),
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

            # If the turn has an attachment, has an url, is from a bot, mentions another channel, or isn't a regular message or reply, mask entire turn
            # Teaching it to output any of this stuff is probably bad, but would probably also be bad contextually to remove all together
            if (
                turn["attachments"]
                or turn["stickers"]
                or re.search(URL_FINDING_REGEX_PATTERN, turn_value)
                or turn["isbot"]
                or "#" in turn_value  # TODO: Find a better way to check if a turn mentions another channel
                or turn["type"] not in {"Default", "Reply"}
            ):
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            else:
                labels = (
                    [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
                    + [*copy.deepcopy(res["input_ids"])][len(prefix["input_ids"]):]
                )

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
class CustomLLaMa3TrashLogsV3Prompter:
    """
    Prompter for CustomLLaMa3TrashLogsV3.
    """

    def __init__(self, *args, **kwargs):
        # Constructor does nothing
        pass


# Function to load the CustomLLaMa3TrashLogsV3PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomLLaMa3TrashLogsV3PromptTokenizingStrategy(
        CustomLLaMa3TrashLogsV3Prompter(),  # TODO: Remove this as it doesn't get used
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )
