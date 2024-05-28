"""LLaMa-3-Instruct prompt tokenization strategy for ORPO"""
from typing import Any, Dict, Generator, List, Optional, Tuple

from pydantic import BaseModel

from axolotl.prompt_tokenizers import IGNORE_INDEX, PromptTokenizingStrategy
from axolotl.prompters import Prompter


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

        # We don't want to remove the BOS token for the first turn
        strip_bos = False

        # Sometimes it gets named 'conversations' and other times 'conversation'
        if "conversations" in prompt:
            conversation_name = "conversations"
        elif "conversation" in prompt:
            conversation_name = "conversation"
        else:
            LOG.warning(f"sample does not contain 'conversations' or 'conversation'")
            exit()

        # Iterate over each conversation turn in the prompt
        for turn in prompt[conversation_name]:
            if "from" in turn and "value" in turn:
                sharegpt_from, sharegpt_value = (
                    turn["from"],
                    turn["value"]
                )

                if sharegpt_from == "system":
                    role_name = "system"
                elif sharegpt_from == "human":
                    role_name = "user"
                elif sharegpt_from == "gpt":
                    role_name = "assistant"
                else:
                    LOG.warning(f"'from' contains an unhandled string")
                    exit()
            else:
                LOG.warning(f"conversation does not contain 'from' or 'value'")
                exit()

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self._tokenize(
                f"<|start_header_id|>{role_name}<|end_header_id|>\n\n",
                add_eos_token=False,
                strip_bos_token=strip_bos,
            )

            # Get entire tokenized turn
            res = self._tokenize(
                f"<|start_header_id|>{role_name}<|end_header_id|>\n\n"
                f"{sharegpt_value.strip()}<|eot_id|>",
                add_eos_token=False,
                strip_bos_token=strip_bos,
            )

            # Handle masked user turn
            if (
                self.train_on_inputs is False
                and (
                    sharegpt_from == "system"
                    or sharegpt_from == "human"
                )
            ):
                input_ids += res
                labels += [IGNORE_TOKEN_ID] * len(res["input_ids"])
            # Handle partially masked model turn
            elif (
                self.train_on_inputs is False
                and sharegpt_from == "gpt"
            ):
                input_ids += res
                labels += (
                    [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
                    + [*copy.deepcopy(res["input_ids"])][len(prefix["input_ids"]):]
                )
            # Handle unmasked turn
            else:
                input_ids += res
                labels += res["input_ids"]

            # Now that we've done the first turn we can remove the BOS token
            strip_bos = True

        # Deal with the chosen response, and combine with main conversation
        prefix = self._tokenize(
            f"<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_eos_token=False,
            strip_bos_token=True,
        )
        res = self._tokenize(
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{prompt['chosen_gpt']}<|eot_id|>",
            add_eos_token=True,
            strip_bos_token=True,
        )
        chosen_input_ids = input_ids + res
        chosen_labels = labels + (
            [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
            + [*copy.deepcopy(res["input_ids"])][len(prefix["input_ids"]):]
        )

        # Deal with the rejected response, and combine with main conversation
        prefix = self._tokenize(
            f"<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_eos_token=False,
            strip_bos_token=True,
        )
        res = self._tokenize(
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{prompt['rejected_gpt']}<|eot_id|>",
            add_eos_token=True,
            strip_bos_token=True,
        )
        rejected_input_ids = input_ids + res
        rejected_labels = labels + (
            [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
            + [*copy.deepcopy(res["input_ids"])][len(prefix["input_ids"]):]
        )

        # print("---chosen_input_ids---")
        # print(chosen_input_ids)
        # print("---chosen_input_ids---")
        # print("---chosen_labels---")
        # print(chosen_labels)
        # print("---chosen_labels---")
        # exit()
        return {
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": [1] * len(rejected_labels),
            "input_ids": chosen_input_ids,
            "labels": chosen_labels,
            "attention_mask": [1] * len(chosen_labels),
            "prompt_attention_mask": (
                ([1] * len(input_ids)) + ([0] * (len(chosen_labels) - len(input_ids)))
            ),  # set 1 for all tokens before the chosen/rejected stuff
        }
