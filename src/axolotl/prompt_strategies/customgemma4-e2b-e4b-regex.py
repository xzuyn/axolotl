"""Module containing the CustomGemma4PromptTokenizingStrategy class"""

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


class CustomGemma4PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomGemma4.
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
        try:
            if self.tokenizer.bos_token_id is not None:
                all_input_ids, all_attention_mask, all_labels = (
                    [self.tokenizer.bos_token_id],
                    [1],
                    [IGNORE_TOKEN_ID]
                )
            else:
                all_input_ids, all_attention_mask, all_labels = [], [], []

            # ShareGPT-to-Gemma4 Dictionary
            role_dict = {
                "system": "system",
                "human": "user",
                "gpt": "model",
                # Extra
                "human-chat": "user",
                "gpt-chat": "model",
                # OpenAI/messages
                "user": "user",
                "assistant": "model",
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
                if turn[from_name] in ["human-chat", "gpt-chat"]:
                    sharegpt_value = ftfy.fix_text(
                        f"{turn['name'].strip()}: {turn[value_name].strip()}"
                    )
                else:
                    sharegpt_value = ftfy.fix_text(turn[value_name].strip())

                # Get string which will be masked out if using train_on_inputs: false
                prefix_text = (
                    "\n" if i != 0 else ""
                ) + f"<|turn>{role_dict[turn[from_name]]}\n"

                # Tokenize and create mask out undesired tokens using regex patterns
                tokenized_text, regex_labels = regex_attention_tokenizer(
                    tokenizer=self.tokenizer,
                    text=f"{prefix_text}{sharegpt_value}<turn|>",
                )

                # Handle masked user turn
                if self.train_on_inputs is False and turn[from_name] in [
                    "system",
                    "user",
                    "human",
                    "human-chat",
                ]:
                    turn_segments.append(
                        {
                            from_name: turn[from_name],
                            "input_ids": tokenized_text["input_ids"],
                            "attention_mask": tokenized_text["attention_mask"],
                            "labels": [IGNORE_TOKEN_ID] * len(regex_labels),
                        }
                    )
                # Handle partially masked model turn
                elif self.train_on_inputs is False and turn[from_name] in [
                    "assistant",
                    "gpt",
                    "gpt-chat",
                ]:
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
                # Handle unmasked turn
                else:
                    turn_segments.append(
                        {
                            from_name: turn[from_name],
                            "input_ids": tokenized_text["input_ids"],
                            "attention_mask": tokenized_text["attention_mask"],
                            "labels": regex_labels,
                        }
                    )

            # Combine all the turn segments
            for turn_segment in turn_segments:
                all_input_ids.extend(turn_segment["input_ids"])
                all_attention_mask.extend(turn_segment["attention_mask"])
                all_labels.extend(turn_segment["labels"])

            # Training on samples with all tokens masked is a waste of compute
            # May be worth checking if less than X% of tokens are trainable too
            if all(label == IGNORE_TOKEN_ID for label in all_labels):
                LOG.warning(
                    f"Processed sample will return empty due to no trainable tokens after masking"
                )
                return {"input_ids": [], "attention_mask": [], "labels": []}

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels,
            }
        except Exception as e:
            LOG.warning(e)
            return {"input_ids": [], "attention_mask": [], "labels": []}


# Function to load the CustomGemma4PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomGemma4PromptTokenizingStrategy(
        None, tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
