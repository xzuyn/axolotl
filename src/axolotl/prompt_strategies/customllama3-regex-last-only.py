"""Module containing the CustomLLaMa3PromptTokenizingStrategy class"""

# Import necessary modules and functions
import re

try:
    import ftfy
except ImportError:
    raise ImportError("You need ftfy. https://pypi.org/project/ftfy/")
import logging

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy

try:
    from axolotl.prompt_strategies.regex_attention import regex_attention_tokenizer
except ImportError:
    raise ImportError(
        "You need https://github.com/xzuyn/axolotl/blob/latest-formatters/src/axolotl/prompt_strategies/regex_attention.py"
    )


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


class CustomLLaMa3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomLLaMa3.
    """

    def __init__(
        self, prompter, tokenizer, train_on_inputs, sequence_len, *args, **kwargs
    ):
        # Call the superclass' constructor
        super().__init__(
            prompter=prompter,
            tokenizer=tokenizer,
            train_on_inputs=train_on_inputs,
            sequence_len=sequence_len,
            *args,
            **kwargs,
        )

    def tokenize_prompt(self, prompt):
        # Some tokenizers don't contain this, so if it doesn't exist assume it is set to True
        add_bos = getattr(self.tokenizer, "add_bos_token", True)

        # ShareGPT-to-LLaMa3 Dictionary
        role_dict = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            # Extra
            "human-chat": "user",
            "gpt-chat": "assistant",
            # OpenAI/messages
            "user": "user",
            "assistant": "assistant",
        }

        if "conversations" in prompt:
            conversation_name = "conversations"
            from_name = "from"
            value_name = "value"
        elif "conversation" in prompt:
            conversation_name = "conversation"
            from_name = "from"
            value_name = "value"
        elif "messages" in prompt:
            conversation_name = "messages"
            from_name = "role"
            value_name = "content"
        else:
            LOG.warning(
                f"sample does not contain 'conversations' or 'conversation' or 'messages'"
            )
            exit()

        # Iterate over each conversation turn in the prompt
        turn_segments = []
        for i, turn in enumerate(prompt[conversation_name]):
            try:
                if turn[from_name] in ["human-chat", "gpt-chat"]:
                    sharegpt_value = (
                        f"{turn['name'].strip()}: {turn[value_name].strip()}"
                    )
                else:
                    sharegpt_value = turn[value_name].strip()
            except AttributeError:
                LOG.warning(f"Processed sample will return empty due to AttributeError")
                return {"input_ids": [], "attention_mask": [], "labels": []}

            # Get string which will be masked out if using train_on_inputs: false
            prefix_text = (
                f"<|start_header_id|>{role_dict[turn[from_name]]}<|end_header_id|>\n\n"
            )

            # Tokenize and create mask out undesired tokens using regex patterns
            tokenized_text, regex_labels = regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=f"{prefix_text}{ftfy.fix_text(sharegpt_value).strip()}<|eot_id|>",
            )

            # Handle partially masked last model turn
            if i == len(prompt[conversation_name]) - 1:
                prefix_token_count = 0
                for start, end in tokenized_text["offset_mapping"]:
                    if end <= len(prefix_text):
                        prefix_token_count += 1
                    else:
                        break

                turn_segments.append(
                    {
                        from_name: turn[from_name],
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": (
                            [IGNORE_TOKEN_ID] * prefix_token_count  # Mask the prefix
                            + regex_labels[prefix_token_count:]
                        ),
                    }
                )
            # Handle masked turn
            else:
                turn_segments.append(
                    {
                        from_name: turn[from_name],
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": [IGNORE_TOKEN_ID] * len(regex_labels),
                    }
                )

        # No logic for dropping turns past sequence_len. Last turn must remain intact

        # Combine all the turn segments
        input_ids, attention_mask, labels = [], [], []
        for turn_segment in turn_segments:
            input_ids.extend(turn_segment["input_ids"])
            attention_mask.extend(turn_segment["attention_mask"])
            labels.extend(turn_segment["labels"])

        # Add missing BOS token if needed
        if (
            add_bos
            and self.tokenizer.bos_token_id
            and input_ids[0] != self.tokenizer.bos_token_id
        ):
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 1)
            labels.insert(0, IGNORE_TOKEN_ID)

        # Training on samples with all tokens masked is a waste of compute
        # May be worth checking if less than X% of tokens are trainable too
        if all(label == IGNORE_TOKEN_ID for label in labels):
            LOG.warning(
                f"Processed sample will return empty due to no trainable tokens after masking"
            )
            return {"input_ids": [], "attention_mask": [], "labels": []}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# Function to load the CustomLLaMa3PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomLLaMa3PromptTokenizingStrategy(
        None, tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
