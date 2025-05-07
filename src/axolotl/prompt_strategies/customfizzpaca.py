"""Module containing the CustomFizzpacaPromptTokenizingStrategy class"""

# Import necessary modules and functions
import ftfy
import logging

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100


class CustomFizzpacaPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomFizzpaca.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)

    def tokenize_prompt(self, prompt):
        # ShareGPT-to-Fizzpaca Dictionary
        role_dict = {
            "system": "### System:",
            "human": "### Instruction:",
            "gpt": "### Response:",
            "human-chat": "### Instruction:",
            "gpt-chat": "### Response:"
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
        input_ids, attention_mask, labels = [], [], []
        for i, turn in enumerate(prompt[conversation_name]):
            if turn["from"] in ["human-chat", "gpt-chat"]:
                sharegpt_value = f"{turn['name'].strip()}: {turn['value'].strip()}"
            else:
                sharegpt_value = turn["value"].strip()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix_text = f"{'\n\n' if i != 0 else ''}{role_dict[turn['from']]}\n"
            tokenized_prefix_text = self.tokenizer(
                text=prefix_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            # Get entire tokenized turn
            eot_text = self.tokenizer.eos_token if turn['from'] in ['gpt', 'gpt-chat'] else ''
            tokenized_text = self.tokenizer(
                text=f"{prefix_text}{ftfy.fix_text(sharegpt_value.strip())}{eot_text}",
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )

            # Handle masked user turn
            if self.train_on_inputs is False and turn["from"] in ["system", "human", "human-chat"]:
                input_ids += tokenized_text["input_ids"]
                attention_mask += tokenized_text["attention_mask"]
                labels += [IGNORE_TOKEN_ID] * len(tokenized_text["input_ids"])
            # Handle partially masked model turn
            elif self.train_on_inputs is False and turn["from"] in ["gpt", "gpt-chat"]:
                input_ids += tokenized_text["input_ids"]
                attention_mask += tokenized_text["attention_mask"]
                labels += (
                    [IGNORE_TOKEN_ID] * len(tokenized_prefix_text["input_ids"])  # Mask the prefix
                    + tokenized_text["input_ids"][len(tokenized_prefix_text["input_ids"]):]
                )
            # Handle unmasked turn
            else:
                input_ids += tokenized_text["input_ids"]
                attention_mask += tokenized_text["attention_mask"]
                labels += tokenized_text["input_ids"]

        # Add missing BOS token
        if self.tokenizer.bos_token_id and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 0)
            labels.insert(0, self.tokenizer.bos_token_id)

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


# Function to load the FizzpacaPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomFizzpacaPromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs)
