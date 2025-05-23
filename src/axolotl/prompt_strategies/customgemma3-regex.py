"""Module containing the CustomGemma3PromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
try:
    import ftfy
except ImportError:
    raise ImportError("You need ftfy. https://pypi.org/project/ftfy/")
import logging
from copy import deepcopy

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


class CustomGemma3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomGemma3.
    """

    def __init__(self, prompter, tokenizer, train_on_inputs, sequence_len, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(
            prompter=prompter,
            tokenizer=tokenizer,
            train_on_inputs=train_on_inputs,
            sequence_len=sequence_len,
            *args,
            **kwargs
        )

    def tokenize_prompt(self, prompt):
        # ShareGPT-to-Gemma3 Dictionary
        role_dict = {
            "system": "system",
            "human": "user",
            "gpt": "model",
            "human-chat": "user",
            "gpt-chat": "model"
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
        turn_segments = []
        for i, turn in enumerate(prompt[conversation_name]):
            try:
                if turn["from"] in ["human-chat", "gpt-chat"]:
                    sharegpt_value = f"{turn['name'].strip()}: {turn['value'].strip()}"
                else:
                    sharegpt_value = turn["value"].strip()
            except AttributeError:
                LOG.warning(f"Processed sample will return empty due to AttributeError")
                return {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": []
                }

            # Get string which will be masked out if using train_on_inputs: false
            prefix_text = ("\n" if i != 0 else "") + f"<start_of_turn>{role_dict[turn['from']]}\n"

            # Tokenize and create mask out undesired tokens using regex patterns
            tokenized_text, regex_mask_labels = mask_regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=f"{prefix_text}{ftfy.fix_text(sharegpt_value).strip()}<end_of_turn>",
                compiled_regex_patterns=COMPILED_REGEX_PATTERNS,
            )

            # Handle masked user turn
            if self.train_on_inputs is False and turn["from"] in ["system", "human", "human-chat"]:
                turn_segments.append(
                    {
                        "from": turn["from"],
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": [IGNORE_TOKEN_ID] * len(regex_mask_labels),
                    }
                )
            # Handle partially masked model turn
            elif self.train_on_inputs is False and turn["from"] in ["gpt", "gpt-chat"]:
                prefix_token_count = 0
                for start, end in tokenized_text["offset_mapping"]:
                    if end <= len(prefix_text):
                        prefix_token_count += 1
                    else:
                        break

                turn_segments.append(
                    {
                        "from": turn["from"],
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": (
                            [IGNORE_TOKEN_ID] * prefix_token_count  # Mask the prefix
                            + regex_mask_labels[prefix_token_count:]
                        ),
                    }
                )
            # Handle unmasked turn
            else:
                turn_segments.append(
                    {
                        "from": turn["from"],
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": regex_mask_labels,
                    }
                )

        # Only keep turns which add up to less than sequence_len (or seq_len - 1 if bos is set)
        current_length = 0
        trimmed_turn_segments = []
        for turn_segment in turn_segments:
            turn_segment_length = len(turn_segment["input_ids"])
            if current_length + turn_segment_length > self.sequence_len - (1 if self.tokenizer.bos_token_id else 0):
                break
            else:
                trimmed_turn_segments.append(turn_segment)
                current_length += turn_segment_length

        # Ensure the final turn is from gpt or gpt-chat
        while trimmed_turn_segments and trimmed_turn_segments[-1]["from"] not in ["gpt", "gpt-chat"]:
            trimmed_turn_segments.pop()

        # Return empty if there are less than 2 turns left
        if len(trimmed_turn_segments) < 2:
            # LOG.warning(f"Processed sample will return empty due to not enough turns")
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }

        # Combine all the turn segments
        input_ids, attention_mask, labels = [], [], []
        for turn_segment in trimmed_turn_segments:
            input_ids.extend(turn_segment["input_ids"])
            attention_mask.extend(turn_segment["attention_mask"])
            labels.extend(turn_segment["labels"])

        # Add missing BOS token if needed
        if self.tokenizer.add_bos_token and self.tokenizer.bos_token_id and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 1)
            labels.insert(0, IGNORE_TOKEN_ID)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# Function to load the CustomGemma3PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomGemma3PromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs, cfg.sequence_len)
