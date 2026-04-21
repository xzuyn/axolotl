"""Module containing the CustomMultiPromptTokenizingStrategy class"""

try:
    import ftfy
except ImportError:
    raise ImportError("You need ftfy. https://pypi.org/project/ftfy/")
import logging
import random

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


class CustomMultiPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomMulti.
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

    def handle_chatml(self, i, role, content):
        role_dict = {
            # ShareGPT
            "system": "system",
            "human": "user",
            "gpt": "model",
            # OpenAI/messages
            "user": "user",
            "assistant": "model",
        }
        prefix_text = (
            "\n" if i != 0 else ""
        ) + f"<|im_start|>{role_dict[role]}\n"
        full_text = f"{prefix_text}{content}<|im_end|>"
        return prefix_text, full_text

    def handle_llama3(self, i, role, content):
        role_dict = {
            # ShareGPT
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            # OpenAI/messages
            "user": "user",
            "assistant": "assistant",
        }
        prefix_text = f"<|start_header_id|>{role_dict[role]}<|end_header_id|>\n\n"
        full_text = f"{prefix_text}{content}<|eot_id|>"
        return prefix_text, full_text

    def handle_gemma3(self, i, role, content):
        role_dict = {
            # ShareGPT
            "system": "system",
            "human": "user",
            "gpt": "model",
            # OpenAI/messages
            "user": "user",
            "assistant": "model",
        }
        prefix_text = (
            "\n" if i != 0 else ""
        ) + f"<start_of_turn>{role_dict[role]}\n"
        full_text = f"{prefix_text}{content}<end_of_turn>"
        return prefix_text, full_text

    def handle_gemma4(self, i, role, content):
        role_dict = {
            # ShareGPT
            "system": "system",
            "human": "user",
            "gpt": "model",
            # OpenAI/messages
            "user": "user",
            "assistant": "model",
        }
        prefix_text = (
            "\n" if i != 0 else ""
        ) + f"<|turn>{role_dict[role]}\n"
        full_text = f"{prefix_text}{content}<turn|>"
        return prefix_text, full_text

    def handle_fizzpaca(self, i, role, content):
        role_dict = {
            # ShareGPT
            "system": "### System:",
            "human": "### Instruction:",
            "gpt": "### Response:",
            # OpenAI/messages
            "user": "### Instruction:",
            "assistant": "### Response:",
        }
        prefix_text = (
            "\n\n" if i != 0 else ""
        ) + f"{role_dict[role]}\n"
        full_text = f"{prefix_text}{content}</s>"
        return prefix_text, full_text

    def handle_mistral(self, i, role, content):
        role_dict = {
            # ShareGPT
            "system": ["[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"],
            "human": ["[INST]", "[/INST]"],
            "gpt": ["", "</s>"],
            # OpenAI/messages
            "user": ["[INST]", "[/INST]"],
            "assistant": ["", "</s>"],
        }
        prefix_text = role_dict[role][0]
        full_text = f"{prefix_text}{content}{role_dict[role][1]}"
        return prefix_text, full_text

    def handle_metharme(self, i, role, content):
        role_dict = {
            # ShareGPT
            "system": "<|system|>",
            "human": "<|user|>",
            "gpt": "<|model|>",
            # OpenAI/messages
            "user": "<|user|>",
            "assistant": "<|model|>",
        }
        prefix_text = role_dict[role][0]
        full_text = f"{prefix_text}{content}</s>"
        return prefix_text, full_text

    def tokenize_prompt(self, prompt):
        try:
            if self.tokenizer.bos_token_id is not None:
                all_input_ids, all_attention_mask, all_labels, all_token_type_ids, all_mm_token_type_ids = (
                    [self.tokenizer.bos_token_id], [1], [IGNORE_TOKEN_ID], [0], [0]
                )
            else:
                all_input_ids, all_attention_mask, all_labels, all_token_type_ids, all_mm_token_type_ids = (
                    [], [], [], [], []
                )

            random_handle = random.choice(
                [
                    self.handle_chatml, self.handle_llama3, self.handle_gemma3,
                    self.handle_gemma4, self.handle_fizzpaca, self.handle_mistral, self.handle_metharme
                ]
            )

            turn_segments = []
            for i, turn in enumerate(prompt["prompt"]):
                _, full_text = random_handle(
                    i=i,
                    role=turn["from"],
                    content=ftfy.fix_text(turn["value"].strip())
                )

                tokenized_text = self.tokenizer(
                    text=full_text,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )

                # All prompt turns are masked
                turn_segments.append(
                    {
                        "input_ids": tokenized_text["input_ids"],
                        "attention_mask": tokenized_text["attention_mask"],
                        "labels": [IGNORE_TOKEN_ID] * len(tokenized_text["input_ids"]),
                        "token_type_ids": [0] * len(tokenized_text["input_ids"]),
                        "mm_token_type_ids": [0] * len(tokenized_text["input_ids"]),
                    }
                )

            prefix_text, full_text = random_handle(
                i=len(prompt["prompt"]),
                role="assistant",
                content=ftfy.fix_text(prompt["response"].strip())
            )

            tokenized_text, regex_labels = regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=full_text,
            )

            prefix_token_count = 0
            for start, end in tokenized_text["offset_mapping"]:
                if end <= len(prefix_text):
                    prefix_token_count += 1
                else:
                    break

            turn_segments.append(
                {
                    "input_ids": tokenized_text["input_ids"],
                    "attention_mask": tokenized_text["attention_mask"],
                    "labels": (
                        [IGNORE_TOKEN_ID] * prefix_token_count  # Mask the prefix
                        + regex_labels[prefix_token_count:]
                    ),
                    "token_type_ids": [0] * len(regex_labels),
                    "mm_token_type_ids": [0] * len(regex_labels),
                }
            )

            # Combine all the turn segments
            for turn_segment in turn_segments:
                all_input_ids.extend(turn_segment["input_ids"])
                all_attention_mask.extend(turn_segment["attention_mask"])
                all_labels.extend(turn_segment["labels"])
                all_token_type_ids.extend(turn_segment["token_type_ids"])
                all_mm_token_type_ids.extend(turn_segment["mm_token_type_ids"])

            # Training on samples with all tokens masked is a waste of compute
            # May be worth checking if less than X% of tokens are trainable too
            if all(label == IGNORE_TOKEN_ID for label in all_labels):
                LOG.warning(
                    f"Processed sample will return empty due to no trainable tokens after masking"
                )
                return {"input_ids": [], "attention_mask": [], "labels": [], "token_type_ids": [], "mm_token_type_ids": []}

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels,
                "token_type_ids": all_token_type_ids,
                "mm_token_type_ids": all_mm_token_type_ids,
            }
        except Exception as e:
            LOG.warning(e)
            return {"input_ids": [], "attention_mask": [], "labels": [], "token_type_ids": [], "mm_token_type_ids": []}


# Function to load the CustomMultiPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomMultiPromptTokenizingStrategy(
        None, tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
