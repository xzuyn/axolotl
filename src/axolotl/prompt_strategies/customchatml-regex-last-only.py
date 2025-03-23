"""Module containing the CustomChatMLPromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
import ftfy
import logging

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy
try:
    from axolotl.prompt_strategies.formatter_regex import COMPILED_REGEX_PATTERNS
except ImportError:
    raise ImportError("You need https://github.com/xzuyn/axolotl/blob/prompt_formats/src/axolotl/prompt_strategies/formatter_regex.py")


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


class CustomChatMLPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomChatML.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)

    def tokenize_prompt(self, prompt):
        # ShareGPT-to-ChatML Dictionary
        role_dict = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            "human-chat": "user",
            "gpt-chat": "assistant"
        }

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
            if turn["from"] == "human-chat":
                sharegpt_value = f"{turn['name'].strip()}: {turn['value'].strip()}"
            elif turn["from"] == "gpt-chat":
                sharegpt_value = f"{turn['name'].strip()}: {turn['value'].strip()}"
            else:
                sharegpt_value = turn["value"].strip()

            sharegpt_value = sharegpt_value.replace("…", "...")

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix_text = (("\n" if i != 0 else "") + f"<|im_start|>{ftfy.fix_text(role_dict[turn['from']]).strip()}\n")
            tokenized_prefix_text = self.tokenizer(
                text=prefix_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            # Tokenize and create mask out undesired tokens using regex patterns
            tokenized_text = mask_regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=f"{prefix_text}{ftfy.fix_text(sharegpt_value.strip())}<|im_end|>",
                compiled_regex_patterns=COMPILED_REGEX_PATTERNS,
            )

            # Handle partially masked last model turn
            if i == len(prompt[conversation_name]) - 1:
                tokenized_text["attention_mask"] = (
                    [0] * len(tokenized_prefix_text["attention_mask"])  # Mask the prefix
                    + tokenized_text["attention_mask"][len(tokenized_prefix_text["attention_mask"]):]
                )
            # Handle masked turn
            else:
                tokenized_text["attention_mask"] = [0] * len(tokenized_text["attention_mask"])

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


# Function to load the CustomChatMLPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomChatMLPromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs)
