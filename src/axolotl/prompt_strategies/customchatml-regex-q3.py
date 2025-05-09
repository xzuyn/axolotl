"""Module containing the CustomChatMLPromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
try:
    import ftfy
except ImportError:
    raise ImportError("You need ftfy. https://pypi.org/project/ftfy/")
import logging
from copy import deepcopy
import random

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy
try:
    from axolotl.prompt_strategies.formatter_regex import COMPILED_REGEX_PATTERNS
except ImportError:
    raise ImportError("You need https://github.com/xzuyn/axolotl/blob/came-plus-formatters/src/axolotl/prompt_strategies/formatter_regex.py")


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

    regex_mask_labels = deepcopy(tokenized_text["input_ids"])
    for pattern in compiled_regex_patterns:
        for match in pattern.finditer(text):
            found_index = match.start()
            end_index = match.end()

            # Check each token's character span; if it overlaps, mask it out.
            for i, (token_start, token_end) in enumerate(tokenized_text["offset_mapping"]):
                if token_start < end_index and token_end > found_index:
                    regex_mask_labels[i] = IGNORE_TOKEN_ID

    return tokenized_text, regex_mask_labels


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

        if prompt[conversation_name][0]["from"] != "system":
            prompt[conversation_name].insert(0, {"from": "system", "value": "/no_think"})
            need_to_fix_system = False
        else:
            need_to_fix_system = True

        # Iterate over each conversation turn in the prompt
        input_ids, attention_mask, labels = [], [], []
        for i, turn in enumerate(prompt[conversation_name]):
            if turn["from"] in ["human-chat", "gpt-chat"]:
                sharegpt_value = f"{turn['name'].strip()}: {turn['value'].strip()}"
            elif turn["from"] == "system" and need_to_fix_system:
                sharegpt_value = random.choice(
                    [
                        f"/no_think {turn['value'].strip()}",
                        f"/no_think\n{turn['value'].strip()}",
                        f"/no_think\n\n{turn['value'].strip()}",
                        f"{turn['value'].strip()} /no_think",
                        f"{turn['value'].strip()}\n/no_think",
                        f"{turn['value'].strip()}\n\n/no_think"
                    ]
                )
            else:
                sharegpt_value = turn["value"].strip()

            if turn["from"] in ["gpt", "gpt-chat"]:
                sharegpt_value = f"<think>\n\n</think>\n\n{sharegpt_value}"

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix_text = ("\n" if i != 0 else "") + f"<|im_start|>{role_dict[turn['from']]}\n"
            tokenized_prefix_text = self.tokenizer(
                text=prefix_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            # Tokenize and create mask out undesired tokens using regex patterns
            tokenized_text, regex_mask_labels = mask_regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=f"{prefix_text}{ftfy.fix_text(sharegpt_value).strip()}<|im_end|>",
                compiled_regex_patterns=COMPILED_REGEX_PATTERNS,
            )

            # Handle masked user turn
            if self.train_on_inputs is False and turn["from"] in ["system", "human", "human-chat"]:
                input_ids += tokenized_text["input_ids"]
                attention_mask += tokenized_text["attention_mask"]
                labels += [IGNORE_TOKEN_ID] * len(regex_mask_labels)
            # Handle partially masked model turn
            elif self.train_on_inputs is False and turn["from"] in ["gpt", "gpt-chat"]:
                input_ids += tokenized_text["input_ids"]
                attention_mask += tokenized_text["attention_mask"]
                labels += (
                    [IGNORE_TOKEN_ID] * len(tokenized_prefix_text["input_ids"])  # Mask the prefix
                    + regex_mask_labels[len(tokenized_prefix_text["input_ids"]):]
                )
            # Handle unmasked turn
            else:
                input_ids += tokenized_text["input_ids"]
                attention_mask += tokenized_text["attention_mask"]
                labels += regex_mask_labels

        # Add missing BOS token
        if self.tokenizer.bos_token_id and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 1)
            labels.insert(0, IGNORE_TOKEN_ID)

        # Add missing EOS token
        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)
            labels.append(self.tokenizer.eos_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# Function to load the CustomChatMLPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomChatMLPromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs)
