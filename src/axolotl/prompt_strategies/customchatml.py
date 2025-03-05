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
        input_ids, attention_mask, labels = [], [], []
        for i, turn in enumerate(prompt[conversation_name]):
            # Get correct roles and messages
            sharegpt_from, sharegpt_value = turn["from"].strip(), turn["value"].strip()

            # ShareGPT Roles
            if sharegpt_from == "system":
                role_name = "system"
            elif sharegpt_from == "human":
                role_name = "user"
            elif sharegpt_from == "gpt":
                role_name = "assistant"
            # CustomShareGPT Roles
            elif sharegpt_from == "human-chat":
                role_name = "user"
                sharegpt_value = f"{turn['name'].strip()}: {sharegpt_value}"
            elif sharegpt_from == "gpt-chat":
                role_name = "assistant"
                sharegpt_value = f"{turn['name'].strip()}: {sharegpt_value}"
            elif sharegpt_from == "thought":
                role_name = "thought"
            else:
                LOG.warning(f"'from' contains an unhandled string: {sharegpt_from}")
                exit()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self.tokenizer(
                text=f"{'\n' if i != 0 else ''}<|im_start|>{role_name}\n",
                truncation=False,
                padding=False,
                return_tensors=None,
            )

            # Get entire tokenized turn
            tokenized_text = self.tokenizer(
                text=(
                    f"{'\n' if i != 0 else ''}<|im_start|>{role_name}\n"
                    f"{ftfy.fix_text(sharegpt_value.strip())}<|im_end|>"
                ),
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )

            # Handle masked user turn
            if self.train_on_inputs is False and sharegpt_from in ["system", "human", "human-chat"]:
                tokenized_text["attention_mask"] = [0] * len(tokenized_text["attention_mask"])
            # Handle partially masked model turn
            elif self.train_on_inputs is False and sharegpt_from in ["gpt", "gpt-chat", "thought"]:
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
                for label, mask in zip(tokenized_text["input_ids"], tokenized_text["attention_mask"])
            ]
        }


# Function to load the CustomChatMLPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomChatMLPromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs)
