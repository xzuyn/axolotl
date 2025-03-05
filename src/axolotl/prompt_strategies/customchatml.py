"""Module containing the CustomChatMLPromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
import ftfy
import logging

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


class CustomChatMLPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomChatML.
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
            # ShareGPT-to-ChatML Dictionary
            role_dict = {
                "system": "system",
                "human": "user",
                "gpt": "assistant",
                "human-chat": "user",
                "gpt-chat": "assistant"
            }

            if turn["from"] == "human-chat":
                sharegpt_value = f"{turn['name'].strip()}: {turn['value'].strip()}"
            elif turn["from"] == "gpt-chat":
                sharegpt_value = f"{turn['name'].strip()}: {turn['value'].strip()}"
            else:
                sharegpt_value = turn["value"].strip()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix_text = f"{'\n' if i != 0 else ''}<|im_start|>{role_dict[turn['from']]}\n"
            prefix = self.tokenizer(
                text=prefix_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            # Get entire tokenized turn
            tokenized_text = self.tokenizer(
                text=(
                    f"{prefix_text}{ftfy.fix_text(sharegpt_value.strip())}<|im_end|>"
                ),
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )

            # Handle masked user turn
            if self.train_on_inputs is False and turn["from"] in ["system", "human", "human-chat"]:
                tokenized_text["attention_mask"] = [0] * len(tokenized_text["attention_mask"])
            # Handle partially masked model turn
            elif self.train_on_inputs is False and turn["from"] in ["gpt", "gpt-chat", "thought"]:
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


# Function to load the CustomChatMLPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomChatMLPromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs)
