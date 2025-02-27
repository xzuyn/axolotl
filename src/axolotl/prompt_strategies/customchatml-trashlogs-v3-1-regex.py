"""Module containing the CustomLLaMa3TrashLogsV3PromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
import copy
import logging
from collections import defaultdict
from typing import Generator, List, Tuple

# Import from axolotl package
from axolotl.prompt_tokenizers import (
    PromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100
URL_FINDING_REGEX_PATTERN = (
    r"\b(?:https?|ftp|smtp):\/\/"  # Word boundary + protocol
    r"([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}"  # Domain name
    r"(:\d{1,5})?"  # Optional port
    r"(\/[a-zA-Z0-9#\/?=&._-]*)?"  # Path, query parameters, or fragments
    r"\b"  # Word boundary to prevent partial matches
)

REGEX_PATTERNS = [
    "haze of pleasure",
    "finds solace in",
    "reveling in the satisfaction",
    "with each breath",
    "a delicate dance",
    "wet flesh",
    "sensitive flesh",
    "\\bministration(|s)\\b",
    "audible pop",
    "rivulets",
    "admit it",
    "the ball is in your court",
    "the game is on",
    "the choice is yours",
    "i don't bite... unless you want me to",
    "half-lidded eyes",
    "(he|she|they) worries (his|her|their) bottom lip",
    "warring with",
    "arousal pooling",
    "take your pleasure",
    "(he|she|they) fiddles with the hem of (his|her|their) (skirt|shirt)",
    "kiss-bruised lips",
    "bruising kiss",
    "despite (himself|herself|themselves|themself)",
    "yours to take",
    "\\bwanton\\b",
    "reckless abandon",
    "torn between",
    "knuckles turning white",
    "grins wickedly",
    "fiery red hair",
    "long lashes",
    "propriety be damned",
    "the world narrows",
    "pupils blown wide with pleasure",
    "chestnut eyes",
    "(he|she|they) grasps your chin and forces you to meet (his|her|their) gaze",
    "(he|she|they) bites your ear",
    "nails raking angry red lines down your back",
    "(her|his) cheeks flaming",
    "cheeks hollowing",
    "stars burst behind (his|her) eyes",
    "inner walls clenching around nothing",
    "puckered hole",
    "wet heat",
    "(he|she) whimpers, biting (his|her) lip",
    "dusky nipples",
    "slick fold(|s)",
    "still lodged deep inside (his|her)",
    "heart, body and soul belong to you",
    "the night is still young",
    "\\.\\.\\.for now\\b",
    "whether you like it not",
    "without waiting for response",
    "however, (its|it is|it's) important",
    "important to remember that",
    "once upon",
    "nestled deep within",
    "an ethereal beauty",
    "breathless and eager",
    "whispering words of passion",
    "soft and gentle",
    "shivers (\\w+\\s+)?down",
    "dance of pleasure",
    "(his|her) sex",
    "sent (shockwaves|shock waves)",
    "in a rhythm",
    "wild abandon",
    "exhausted and spent",
    "life would never be the same again",
    "like an electric shock",
    "threatens to consume",
    "what (seemed|felt) like an eternity",
    "(lay|lie) ahead",
    "\\bwet pop\\b",
    "maybe, just maybe",
    "perhaps, just perhaps",
    "starts to blur",
    "but it felt like",
    "unfamiliar, yet",
    "moist fold(|s)",
    "the night is still young",
    "our shared experiences",
    "bond(|s) built on mutual trust",
    "the ball is in your court",
    "little did (he|she|they) know",
    "a pregnant silence",
    "beats like a (\\w+\\s+)?drum",
    "\\bpert\\b",
    "for the sake of keeping things",
    "her breasts heaving with desire",
    "dickick",
    "\\brivulets\\b",
    "arousal pooling in (his|her|their) belly",
    "steeling (her|him)self",
    "the din of the crowd",
    "journey of mutual understanding",
    "revulsion warred with (reluctant|reluctance)",
    "her bare mound(|s)",
    "pooled around her (ankles|feet)",
    "straddles your (waist|lap)",
    "words turn into a purr",
    "grips like a vice",
    "shivers running up",
    "arched spine",
    "penetrated to the hilt",
    "the pressure in (her|his) loins",
    "\\bcunny\\b",
    "catch my drift",
    "\\bloli\\b",
    "sway(|s) hypnotically",
    "tantalizing promise",
    "with each slow, deliberate movement",
    "for what (felt|seemed) like (hours|an eternity|forever)",
    ", but (he|she|they|I) can't help it",
    "conspiratorial whisper(|s)",
    "whisper(|ing) conspiratorially",
    "eyes (sparkling|twinkling) with mischief",
    "\\bministration\\b",
    "couldn(')?t help but",
    "racing with anticipation",
    "leaves little to the imagination",
    "shivers up",
    "waggles her eyebrows",
    "a testament to",
    "a moth to a flame",
    "canvas",
    "eyes glint(ed)?",
    "camaraderie",
    "humble abode",
    "cold and calculating",
    "unbeknownst to them",
    "iridescent",
    "a dance as old as time",
    "husky whispers",
    "seductive purrs",
    "towers over",
    "rich tapestry",
    "delve",
    "lean in",
    "leans in",
    "leans in close",
    "don't stop, don't ever stop",
    "make me yours, claim me",
    "mind, body, and soul",
    "another day in your life",
    "a symphony of",
    "body and soul",
    "pleasure and pain",
    "like a predator stalking its prey",
    "orchestra",
    "depths",
    "dance",
    "chuckles darkly",
    "could not help but",
    "felt a mix of \\w+ and \\w+",
    "a mischievous glint",
    "husky voice",
    "a smirk playing on (her|his|their) lips",
    "playfully smirking",
    "rivulets of",
    "waves of arousal pooling in (her|his|their) belly",
    "torn between propriety and desire",
    "tongue darts out",
    "nails rake angry red lines",
    "(her|his) cheeks flame",
    "inner walls clenching",
    "eyes alight with mirth",
    "naughty boy",
    "tracing a finger along your jawline",
    "oh my\\.\\.\\.",
    "the atmosphere",
    "pushing aside a strand of hair",
    "adam's apple bobbing",
    "\\bpalpable\\b",
    "bosomy breasts",
    "looking like the cat that got the cream",
    "softly but firmly",
    "siren(')?s call",
    "dimly lit",
    "the air is thick",
    "\\\"you really know how to treat a lady!\\\"",
    "\\bpebbled\\b",
    "eyes searching",
    "what do you say",
    "to meet (her|his) gaze",
    "(her|his|their) wet heat",
    "whimpers, biting (her|his|their) lip",
    "hum with delight",
    "embark on this",
    ", if you will,",
    "evident in (his|her|their) eyes",
    "overwhelmed by the sheer",
    "was taken aback",
    "(her|his) cheeks redden",
    "little mouse",
    "\\bminx\\b",
    "you(')?re a bold one"
]
COMPILED_REGEX_PATTERNS = [re.compile(pattern) for pattern in REGEX_PATTERNS]


def mask_regex_attention(self, input_data, compiled_regex_patterns):
    # Decode the input_ids back to text.
    input_text = self.tokenizer.decode(input_data["input_ids"])

    # Re-tokenize the text with offset mapping using the same options as the original tokenization.
    encoded = self.tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoded["offset_mapping"]

    # Make a copy of the original attention_mask.
    new_attention_mask = input_data["attention_mask"].copy()

    # For each regex pattern, find all its occurrences in the text.
    for pattern in compiled_regex_patterns:
        for match in pattern.finditer(input_text):
            found_index = match.start()
            end_index = match.end()
            # Check each token's character span; if it overlaps, mask it out.
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start < end_index and token_end > found_index:
                    new_attention_mask[i] = 0

    return new_attention_mask


class CustomLLaMa3TrashLogsV3PromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for CustomLLaMa3TrashLogsV3.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        # Call the superclass' constructor
        super().__init__(prompter, tokenizer, *args, **kwargs)

    def mask_sensitive_attention(self, input_data, sensitive_strings):
        # Decode the input_ids back to text.
        # Using skip_special_tokens=True to avoid potential extra tokens in the decoded text.
        input_text = self.decode(input_data["input_ids"])

        # Re-tokenize the text with offset mapping using the same options as the original tokenization.
        encoded = self(input_text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = encoded["offset_mapping"]

        # Make a copy of the original attention_mask.
        new_attention_mask = input_data["attention_mask"].copy()

        # For each sensitive string, find all its occurrences in the text.
        for sensitive in sensitive_strings:
            start_search = 0
            while True:
                found_index = input_text.find(sensitive, start_search)
                if found_index == -1:
                    break
                end_index = found_index + len(sensitive)
                # Check each token's character span; if it overlaps, mask it out.
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start < end_index and token_end > found_index:
                        new_attention_mask[i] = 0
                start_search = found_index + 1  # Continue searching after this match.

        # Update the input_data with the modified attention_mask.
        input_data["attention_mask"] = new_attention_mask

        return input_data

    def tokenize_prompt(self, prompt):
        # Tokenize the prompt based on its conversations
        result, current_len = tokenize_prompt_default()

        # Sometimes it gets named 'conversations' and other times 'conversation'
        if "conversations" in prompt:
            conversation_name = "conversations"
        elif "conversation" in prompt:
            conversation_name = "conversation"
        else:
            LOG.warning(f"sample does not contain 'conversations' or 'conversation'")
            exit()

        # Iterate over each conversation turn in the prompt
        num_turns = len(prompt[conversation_name])
        for i, turn in enumerate(prompt[conversation_name]):
            # Strip BOS token and add a new line to the beginning if it's not the first turn
            if i == 0:
                strip_bos = False
                add_new_line = ""
            else:
                strip_bos = True
                add_new_line = "\n"

            # Check if this is the last turn, so we know to add the EOS token
            if i == num_turns - 1:
                end_of_text = True
            else:
                end_of_text = False

            # Add attachment info if it exists
            if turn["attachments"]:
                turn_value = (
                    f"[Attachments: {', '.join(attachment for attachment in turn['attachments'])}]\n\n"
                    f"{turn['value']}"
                ).strip()
            elif turn["stickers"]:
                turn_value = (
                    f"[Stickers: {', '.join(sticker for sticker in turn['stickers'])}]\n\n"
                    f"{turn['value']}"
                ).strip()
            else:
                turn_value = turn["value"].strip()

            turn_from = f"messageid: {turn['messageid']} | timestamp: {turn['timestamp']} | username: {turn['name']} | nickname: {turn['nickname']} | type: {turn['type']}"
            if turn["mentions"]:
                turn_from += f" | mentions: {', '.join(mention for mention in turn['mentions'])}"
            if turn["type"] == "Reply":
                turn_from += f" | reference: {turn['reference']}"

            # Get tokens which will be masked out if using train_on_inputs: false
            prefix = self.tokenizer(
                (
                    f"{add_new_line}<|im_start|>{turn_from}\n"
                ),
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            if prefix["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos:
                prefix["input_ids"] = prefix["input_ids"][1:]
                prefix["attention_mask"] = prefix["attention_mask"][1:]

            # Get entire tokenized turn
            res = self.tokenizer(
                (
                    f"{add_new_line}<|im_start|>{turn_from}\n"
                    f"{turn_value}<|im_end|>"
                ),
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            if res["input_ids"][-1] != self.tokenizer.eos_token_id and end_of_text:
                res["input_ids"].append(self.tokenizer.eos_token_id)
                res["attention_mask"].append(1)
            if res["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos:
                res["input_ids"] = res["input_ids"][1:]
                res["attention_mask"] = res["attention_mask"][1:]

            # If the turn has an attachment, has an url, is from a bot, mentions another channel, or isn't a regular message or reply, mask entire turn
            # Teaching it to output any of this stuff is probably bad, but would probably also be bad contextually to remove all together
            if (
                turn["attachments"]
                or turn["stickers"]
                or re.search(URL_FINDING_REGEX_PATTERN, turn_value)
                or turn["isbot"]
                or "#" in turn_value  # TODO: Find a better way to check if a turn mentions another channel
                or turn["type"] not in {"Default", "Reply"}
            ):
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            else:
                res["attention_mask"] = mask_regex_attention(self, res, COMPILED_REGEX_PATTERNS)
                modified_label = [
                    label if mask == 1
                    else IGNORE_TOKEN_ID
                    for label, mask in zip(res["input_ids"], res["attention_mask"])
                ]
                labels = (
                    [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
                    + modified_label[len(prefix["input_ids"]):]
                )

            # Parse tokenized result and update current length
            result, current_len = parse_tokenized_to_result(
                result,
                current_len,
                res,
                labels,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return result


# TODO: Remove this as it doesn't get used
class CustomLLaMa3TrashLogsV3Prompter:
    """
    Prompter for CustomLLaMa3TrashLogsV3.
    """

    def __init__(self, *args, **kwargs):
        # Constructor does nothing
        pass


# Function to load the CustomLLaMa3TrashLogsV3PromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomLLaMa3TrashLogsV3PromptTokenizingStrategy(
        CustomLLaMa3TrashLogsV3Prompter(),  # TODO: Remove this as it doesn't get used
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )
