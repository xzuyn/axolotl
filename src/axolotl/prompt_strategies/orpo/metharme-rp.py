"""Metharme prompt tokenization strategy for RP ORPO"""
from typing import Any, Dict, Generator, List, Optional, Tuple

from pydantic import BaseModel

from axolotl.prompt_tokenizers import IGNORE_INDEX, PromptTokenizingStrategy
from axolotl.prompters import Prompter


# TODO: Remove/reduce this. We just need ORPOTokenizingStrategy.
def load(
    tokenizer,
    cfg,
    ds_cfg: Optional[Dict[str, Any]] = None, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    return ORPOTokenizingStrategy(
        "placeholder",
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )


# This is doing all the work pretty much.
class ORPOTokenizingStrategy(PromptTokenizingStrategy):
    """
    rejected_input_ids
    input_ids
    rejected_attention_mask
    attention_mask
    rejected_labels
    labels
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def tokenize_prompt(self, prompt):
        input_ids = []
        labels = []

        # Deal with main conversation stuff
        needs_bos = True
        for message in prompt["conversations"]:
            # GPT needs to be handled differently.
            if message["from"] == "gpt-chat":
                part = f"\n{message['name']}: {message['value']}</s>"

                _input_ids = self.tokenizer.encode(part, add_special_tokens=False)
                input_ids += _input_ids

                labels += _input_ids
            else:
                if message["from"] == "system" and needs_bos:
                    part = f"<s><|system|>{message['value']}<|model|>"
                    needs_bos = False
                elif message["from"] == "system" and not needs_bos:
                    part = f"<|system|>{message['value']}<|model|>"
                elif message["from"] == "human-chat":
                    part = f"\n{message['name']}: {message['value']}"
                elif message["from"] == "human":
                    print("metharme-rp only supports system then human-chat and gpt-chat.")
                    exit()
                elif message["from"] == "gpt":
                    print("metharme-rp only supports system then human-chat and gpt-chat.")
                    exit()
                else:
                    continue

                _input_ids = self.tokenizer.encode(part, add_special_tokens=False)
                input_ids += _input_ids
                labels += [IGNORE_INDEX] * len(_input_ids)

        # Deal with the chosen response, and combine with main conversation
        chosen = f"\n{prompt['gpt-name']}{prompt['chosen_gpt']}</s>"
        _input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        chosen_input_ids = input_ids + _input_ids
        chosen_labels = labels + _input_ids

        # Deal with the rejected response, and combine with main conversation
        rejected = f"\n{prompt['gpt-name']}{prompt['rejected_gpt']}</s>"
        _input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)
        rejected_input_ids = input_ids + _input_ids
        rejected_labels = labels + _input_ids

        return {
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": [1] * len(rejected_labels),
            "input_ids": chosen_input_ids,
            "labels": chosen_labels,
            "attention_mask": [1] * len(chosen_labels),
            "prompt_attention_mask": (
                ([1] * len(rejected_input_ids)) + ([0] * (len(chosen_labels) - len(rejected_input_ids)))
            ),
        }
