"""chatml prompt tokenization strategy for ORPO"""
from typing import Any, Dict, Generator, List, Optional, Tuple

from pydantic import BaseModel

from axolotl.prompt_tokenizers import IGNORE_INDEX, PromptTokenizingStrategy
from axolotl.prompters import Prompter
from axolotl.utils.chat_templates import chat_templates


class Message(BaseModel):
    """message/turn"""

    role: str
    content: str
    label: Optional[bool] = None


class MessageList(BaseModel):
    """conversation"""

    messages: List[Message]


def load(
    tokenizer,
    cfg,
    ds_cfg: Optional[Dict[str, Any]] = None, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    chatml transforms for datasets with system, input, chosen, rejected
    """

    # TODO: Remove this. It's not needed anymore.
    chat_template = chat_templates("chatml")
    if ds_cfg and "chat_template" in ds_cfg:
        chat_template = ds_cfg["chat_template"]
        try:
            chat_template = chat_templates(chat_template)
        except ValueError:
            pass
    tokenizer.chat_template = chat_template

    return ORPOTokenizingStrategy(
        ORPOPrompter(chat_template, tokenizer),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        dataset_parser=ORPODatasetParsingStrategy(),
    )


class ORPODatasetParsingStrategy:
    """Strategy to parse chosen rejected dataset into messagelist"""

    # noinspection PyMethodMayBeStatic
    def get_chosen_conversation_thread(self, prompt) -> MessageList:
        """Dataset structure mappings"""

        messages: List[Message] = []
        for i, prompt_message in enumerate(prompt["chosen"]):
            if prompt_message["role"] == "system":
                messages.append(Message(role="system", content=prompt_message["content"], label=False))
            elif prompt_message["role"] == "user":
                messages.append(Message(role="user", content=prompt_message["content"], label=False))
            elif prompt_message["role"] == "gpt":
                messages.append(Message(role="gpt", content=prompt_message["content"], label=True))
        return MessageList(messages=messages)

    # noinspection PyMethodMayBeStatic
    def get_rejected_conversation_thread(self, prompt) -> MessageList:
        """Dataset structure mappings"""

        messages: List[Message] = []
        for i, prompt_message in enumerate(prompt["rejected"]):
            if prompt_message["role"] == "system":
                messages.append(Message(role="system", content=prompt_message["content"], label=False))
            elif prompt_message["role"] == "user":
                messages.append(Message(role="user", content=prompt_message["content"], label=False))
            elif prompt_message["role"] == "gpt":
                messages.append(Message(role="gpt", content=prompt_message["content"], label=True))
        return MessageList(messages=messages)


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
        dataset_parser=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_parser = dataset_parser

    def tokenize_prompt(self, prompt):
        model_tag_ids = self._tokenize("<|model|>", add_eos_token=False, strip_bos_token=True)["input_ids"]
        model_tag_ids_w_bos = self._tokenize("<s><|model|>", add_eos_token=False, strip_bos_token=True)["input_ids"]
        input_ids = []
        labels = []

        # Deal with main conversation stuff
        needs_bos = True
        for message in prompt["conversations"]:
            # GPT needs to be handled differently to mask the model tag.
            if message["from"] == "gpt":
                if needs_bos:
                    part = f"<s><|model|>{message['value']}</s>"
                else:
                    part = f"<|model|>{message['value']}</s>"

                _input_ids = self.tokenizer.encode(part, add_special_tokens=False)
                input_ids += _input_ids
                if needs_bos:
                    labels += ([IGNORE_INDEX] * len(model_tag_ids_w_bos)) + _input_ids[len(model_tag_ids_w_bos):]
                    needs_bos = False
                else:
                    labels += ([IGNORE_INDEX] * len(model_tag_ids)) + _input_ids[len(model_tag_ids):]
            else:
                if message["from"] == "system" and needs_bos:
                    part = f"<s><|system|>{message['value']}"
                    needs_bos = False
                elif message["from"] == "system" and not needs_bos:
                    part = f"<|system|>{message['value']}"
                elif message["from"] == "user" and needs_bos:
                    part = f"<s><|user|>{message['value']}"
                    needs_bos = False
                elif message["from"] == "user" and not needs_bos:
                    part = f"<|user|>{message['value']}"

                _input_ids = self.tokenizer.encode(part, add_special_tokens=False)
                input_ids += _input_ids
                labels += [IGNORE_INDEX] * len(_input_ids)

        # Deal with the chosen response, and combine with main conversation
        chosen = f"<|model|>{prompt['chosen_gpt']}</s>"
        _input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        chosen_input_ids = input_ids + _input_ids
        chosen_labels = labels + (([IGNORE_INDEX] * len(model_tag_ids)) + _input_ids[len(model_tag_ids):])

        # Deal with the rejected response, and combine with main conversation
        rejected = f"<|model|>{prompt['rejected_gpt']}</s>"
        _input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)
        rejected_input_ids = input_ids + _input_ids
        rejected_labels = labels + (([IGNORE_INDEX] * len(model_tag_ids)) + _input_ids[len(model_tag_ids):])


        return {
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": [1] * len(rejected_labels),
            "input_ids": chosen_input_ids,
            "labels": chosen_labels,
            "attention_mask": [1] * len(chosen_labels),
            "prompt_attention_mask": (
                [1] * len(rejected_input_ids) + [0] * (len(chosen_labels) - len(rejected_input_ids))
            ),
        }


# TODO: Remove this. It's not needed anymore.
class ORPOPrompter(Prompter):
    """Single Turn prompter for ORPO"""

    def __init__(self, chat_template, tokenizer):
        self.chat_template = chat_template
        self.tokenizer = tokenizer

    def build_prompt(
        self,
        message_list: MessageList,
    ) -> Generator[Tuple[str, bool], None, None]:
        conversation = []
        for message in message_list.messages:
            conversation.append(message.model_dump())
            if message.role == "system":
                yield self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    chat_template=self.chat_template,
                    tokenize=False,
                ), False
            if message.role == "user":
                yield self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    chat_template=self.chat_template,
                    tokenize=False,
                ), False
            if message.role == "gpt":
                yield self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    chat_template=self.chat_template,
                    tokenize=False,
                ), True
