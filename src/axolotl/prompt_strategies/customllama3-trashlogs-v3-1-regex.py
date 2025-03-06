"""Module containing the CustomLLaMa3TrashLogsV3PromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
import ftfy
import logging
from typing import List, Tuple, Pattern, Dict, Union

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy
try:
    from axolotl.prompt_strategies.formatter_regex import COMPILED_REGEX_PATTERNS
except ImportError:
    print("You need https://github.com/xzuyn/axolotl/blob/prompt_formats/src/axolotl/prompt_strategies/formatter_regex.py")


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


def mask_regex_attention_tokenizer(tokenizer, text, compiled_regex_patterns, add_special_tokens=False):
    tokenized_text = tokenizer(
        text=text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
        return_tensors=None,
        return_offsets_mapping=True,
    )

    for pattern in compiled_regex_patterns:
        for match in pattern.finditer(text):
            found_index = match.start()
            end_index = match.end()

            # Check each token's character span; if it overlaps, mask it out.
            for i, (token_start, token_end) in enumerate(tokenized_text["offset_mapping"]):
                if token_start < end_index and token_end > found_index:
                    tokenized_text["attention_mask"][i] = 0

    return tokenized_text


class CustomLLaMa3TrashLogsV3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomLLaMa3TrashLogsV3.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)

    def tokenize_prompt(self, prompt):
        # Sometimes it gets named 'conversations' and other times 'conversation'
        if "conversations" in prompt:
            conversation_name = "conversations"
        elif "conversation" in prompt:
            conversation_name = "conversation"
        else:
            LOG.warning(f"sample does not contain 'conversations' or 'conversation'")
            exit()

        # Iterate over each conversation turn in the prompt
        input_ids, attention_mask = [], []
        for i, turn in enumerate(prompt[conversation_name]):
            # Add attachment info if it exists
            if turn["attachments"]:
                turn_value = (
                    f"[Attachments: {', '.join(attachment for attachment in turn['attachments'])}]\n\n"
                    f"{turn['value']}"
                ).strip()
            elif turn["stickers"]:
                turn_value = (
                    f"[Stickers: {', '.join(sticker for sticker in turn['stickers'])}]\n\n"
                    f"{turn['value']}"
                ).strip()
            else:
                turn_value = turn["value"].strip()

            turn_from = f"messageid: {turn['messageid']} | timestamp: {turn['timestamp']} | username: {turn['name']} | nickname: {turn['nickname']} | type: {turn['type']}"
            if turn["mentions"]:
                turn_from += f" | mentions: {', '.join(mention for mention in turn['mentions'])}"
            if turn["type"] == "Reply":
                turn_from += f" | reference: {turn['reference']}"

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self.tokenizer(
                text=f"<|start_header_id|>{ftfy.fix_text(turn_from)}<|end_header_id|>\n\n",
                truncation=False,
                padding=False,
                return_tensors=None,
            )

            # Get entire tokenized turn
            tokenized_text = self.tokenizer(
                text=(
                    f"<|start_header_id|>{ftfy.fix_text(turn_from)}<|end_header_id|>\n\n"
                    f"{ftfy.fix_text(turn_value.strip())}<|eot_id|>"
                ),
                truncation=False,
                padding=False,
                return_tensors=None,
                return_offsets_mapping=True,
            )

            # Mask out undesired tokens using regex patterns
            tokenized_text = mask_regex_attention(
                text=(
                    f"<|start_header_id|>{ftfy.fix_text(turn_from)}<|end_header_id|>\n\n"
                    f"{ftfy.fix_text(turn_value.strip())}<|eot_id|>"
                ),
                input_ids=tokenized_text["input_ids"],
                attention_mask=tokenized_text["attention_mask"],
                offset_mapping=tokenized_text["offset_mapping"],
                compiled_regex_patterns=COMPILED_REGEX_PATTERNS,
            )

            # If the turn has an attachment, is from a bot, mentions another channel, or isn't a regular message or reply, mask entire turn
            # Teaching it to output any of this stuff is probably bad, but would probably also be bad contextually to remove all together
            if (
                turn["attachments"]
                or turn["stickers"]
                or turn["isbot"]
                or "#" in turn_value  # TODO: Find a better way to check if a turn mentions another channel
                or turn["type"] not in {"Default", "Reply"}
            ):
                tokenized_text["attention_mask"] = [0] * len(tokenized_text["attention_mask"])
                tokenized_text["labels"] = [IGNORE_TOKEN_ID] * len(tokenized_text["input_ids"])
            else:
                tokenized_text["attention_mask"] = (
                    [0] * len(prefix["attention_mask"])  # Mask the prefix
                    + tokenized_text["attention_mask"][len(prefix["attention_mask"]):]
                )

            input_ids += tokenized_text["input_ids"]
            attention_mask += tokenized_text["attention_mask"]

        # Add missing BOS token
        if self.tokenizer.bos_token_id and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 0)

        # Add missing EOS token
        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [
                label if mask == 1 else IGNORE_TOKEN_ID
                for label, mask in zip(input_ids, attention_mask)
            ]
        }


# Function to load the CustomLLaMa3TrashLogsV3PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomLLaMa3TrashLogsV3PromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs)
