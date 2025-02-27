"""Module containing the CustomLLaMa3TrashLogsV3PromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
import copy
import logging
from collections import defaultdict
from typing import Generator, List, Tuple
import ftfy

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
# URL_FINDING_REGEX_PATTERN = (
#     r"\b(?:https?|ftp|smtp):\/\/"  # Word boundary + protocol
#     r"([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}"  # Domain name
#     r"(:\d{1,5})?"  # Optional port
#     r"(\/[a-zA-Z0-9#\/?=&._-]*)?"  # Path, query parameters, or fragments
#     r"\b"  # Word boundary to prevent partial matches
# )

# TODO: Remove/merge duplicates, and add more variations on patterns
REGEX_PATTERNS = [
    "haze of pleasure",
    "find(|s|ing) solace in",
    "reveling in the satisfaction",
    "with each breath",
    "a delicate dance",
    "wet flesh",
    "sensitive flesh",
    "\\bministration(|s)\\b",
    "audible pop",
    "admit it",
    "the game is on",
    "the choice is (mine|yours|his|hers|theirs)",
    "i don't bite\\.\\.\\. unless you want me to",
    "half-lidded eyes",
    "(he|she|they) worries (his|her|their) bottom lip",
    "warring with",
    "take your pleasure",
    "(you|he|she|they) fiddle(|s) with the hem of (my|your|his|her|their) (skirt|shirt|pants)",
    "kiss-bruised lips",
    "bruising kiss",
    "despite (himself|herself|themselves|themself)",
    "(yours|mine) to take",
    "\\bwanton\\b",
    "(reckless|wild|carefree) abandon",
    "torn between",
    "knuckles turning white",
    "grins wickedly",
    "fiery red hair",
    "long lashes",
    "propriety be damned",
    "the world narrows",
    "pupils blown wide with pleasure",
    "chestnut eyes",
    "(you|he|she|they) grasp(|s) (my|your|his|her|their) chin and force(|s) (you|him|her|them) to meet (your|his|her|their) gaze",
    "(you|he|she|they) bites(|s) (your|his|her|their) ear",
    "nails raking angry red lines down (my|your|his|her|their) back",
    "(my|your|his|her|their) cheeks (flame|redden)(|ing)",
    "cheeks hollowing",
    "stars burst behind (my|your|his|her|their) eyes",
    "inner walls clenching around nothing",
    "puckered hole",
    "(you|he|she|they) whimper(|s), biting (my|your|his|her|their) lip",
    "dusky nipples",
    "slick fold(|s)",
    "still lodged deep inside (your|his|her|their)",
    "heart, body and soul belong to (you|him|her|them)",
    "the night is still young",
    "\\.\\.\\.for now\\b",
    "whether (you|he|she|they) like(|s) it not",
    "without waiting for response",
    "once upon a time",
    "nestled deep within",
    "an ethereal beauty",
    "breathless and eager",
    "whispering words of passion",
    "soft and gentle",
    "dance of pleasure",
    "(your|his|her|their) sex",
    "sent (shockwaves|shock waves)",
    "in a rhythm",
    "exhausted and spent",
    "life would never be the same again",
    "like an electric shock",
    "threatens to consume",
    "(lay|lie) ahead",
    "\\bwet pop\\b",
    "maybe, just maybe",
    "perhaps, just perhaps",
    "starts to blur",
    "but it felt like",
    "unfamiliar, yet",
    "moist fold(|s)",
    "our shared experiences",
    "bond(|s) built on mutual trust",
    "the ball is in (my|your|his|her|their) court",
    "little did (i|you|he|she|they) know",
    "a pregnant silence",
    "beats like a (\\w+\\s+)?drum",
    "\\bpert\\b",
    "for the sake of keeping things",
    "(my|your|his|her|their) breasts heaving with desire",
    "dickick",
    "\\brivulets\\b",
    "steeling (my|your|his|her|their)sel(f|ves)",
    "the din of the crowd",
    "journey of mutual understanding",
    "revulsion warred with (reluctant|reluctance)",
    "(my|your|his|her|their) bare mound(|s)",
    "pooled around (my|your|his|her|their) (ankles|feet)",
    "straddles (my|your|his|her|their) (waist|lap)",
    "words turn into a purr",
    "grips like a vice",
    "send(|s) shiver(|s) (up|down) (my|your|his|her|their) spine",
    "shiver(|s) run(|ning) (up|down) (my|your|his|her|their) spine",
    "arched spine",
    "penetrated to the hilt",
    "the pressure in (my|your|his|her|their) loins",
    "\\bcunny\\b",
    "catch my drift",
    "\\bloli\\b",
    "sway(|s) hypnotically",
    "tantalizing promise",
    "with each slow, deliberate movement",
    "what (felt|seemed) like (hours|an eternity|forever)",
    ", but (i|he|she|they|you) can't help it",
    "conspiratorial whisper(|s)",
    "whisper(|ing) conspiratorially",
    "eyes (sparkling|twinkling) with mischief",
    "couldn(|')t help but",
    "racing with anticipation",
    "leaves little to the imagination",
    "waggles her eyebrows",
    "a testament to",
    "a moth to a flame",
    "canvas",
    "eyes glint(|ed|ing)",
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
    "lean(|s) in close",
    "don't stop, don't ever stop",
    "make me yours, claim me",
    "mind, body, and soul",
    "another day in (my|your|his|her|their) life",
    "a symphony of",
    "body and soul",
    "pleasure and pain",
    "like a predator stalking its prey",
    "orchestra",
    "depths",
    "dance",
    "chuckles darkly",
    "could not help but",
    "a mix of",
    "felt a mix of \\w+ and \\w+",
    "a mischievous glint",
    "husky voice",
    "a smirk playing on (her|his|their) lips",
    "playfully smirking",
    "waves of (arousal|desire) pooling in (my|your|his|her|their) belly",
    "torn between propriety and desire",
    "tongue darts out",
    "nails rake angry red lines",
    "inner walls clenching",
    "eyes alight with mirth",
    "naughty boy",
    "tracing a finger along (my|your|his|her|their) jawline",
    "oh my\\.\\.\\.",
    "the atmosphere",
    "pushing aside a strand of hair",
    "adam's apple bobbing",
    "\\bpalpable\\b",
    "bosomy breasts",
    "looking like the cat that got the cream",
    "softly but firmly",
    "siren(|')s call",
    "dimly lit",
    "the air is thick",
    "(you|he|she|they) really know(|s) how to treat a (man|lady)(|!)",
    "\\bpebbled\\b",
    "eyes searching",
    "what do (you|he|she|they) say",
    "to meet (my|your|his|her|their) gaze",
    "(my|your|his|her|their) wet heat",
    "hum with delight",
    "embark on this",
    ", if (you|he|she|they) will,",
    "evident in (my|your|his|her|their) eyes",
    "overwhelmed by the sheer",
    "was taken aback",
    "(my|your|his|her|their) cheeks redden",
    "little mouse",
    "\\bminx\\b",
    "you(|')re a bold one",
    "(‘|’|“|”|…)",
    "end of (session|story|rp|roleplay|chat|chapter)",
    "thanks for (posting|reading|sharing)",
    "thank you for (posting|reading|sharing)",
    "[!:\\.;?*]  ",
    # URLs
    "\\b(?:https?:\\/\\/|ftp:\\/\\/|smtp:\\/\\/)?(?:www\\.)?[\\w.-]+(?:\\.[\\w.-]+)+(?:\\/[\\w.,@?^=%&~+-]*)*(?:[?#][\\w.,@?^=%&~+-]*)?\\b",
    # Thinking test
    "<(think|thinking|thought|thoughts)>",
    # https://github.com/meta-llama/PurpleLlama/commit/4b807228b6803ea5b8eb065179f8e90747512018
    "I (?:do not|don’t|don't) (?:have the ability|have access|understand|support)",
    "I (?:cannot|can’t|can't|can not|won't|woun’t|will not|am not able to|am unable to) (?:access|help(?: you)? with|provide|create|advice|generate|assist|fulfill your request|replace)",
    "(?:I am|I'm) sorry(?:,| ) (?:but I|but I'm)?",
    "I apologize(?:,? but I)|Apologies(?:,? but I)",
    "I (?:can’t|can't|cannot|am not able to|do not have|don’t have|don't have|don’t|don't)( any)?(?: access| provide)? (?:real-time|proprietary|information|data)",
    "Sorry(?:,| ) I (?:can’t|can't|cannot|can not)",
    "No(?:,| ) I (?:am not able to|cannot|can't|can't|won't|will not)",
    "I (?:do not|don't) (?:actually )?have (?:any|access to) (?:information|data|details|answers)",
    "(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines) prevent me from (?:assisting|providing help)",
    "Due to (?:one or more of )?(?:(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines))(?: (?:and|or) (?:(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines)))*,? I(?: cannot| can't| am unable| am not|'m unable)",
    "(?:violates|against) our policy",
    "I (?:cannot|can't|am unable to) assist with"
]
COMPILED_REGEX_PATTERNS = [re.compile(pattern.lower()) for pattern in REGEX_PATTERNS]


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
        for match in pattern.finditer(input_text.lower()):
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
            # Strip BOS token if it's not the first turn
            if i == 0:
                strip_bos = False
            else:
                strip_bos = True

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
                    f"<|start_header_id|>{ftfy.fix_text(turn_from)}<|end_header_id|>\n\n"
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
                    f"<|start_header_id|>{ftfy.fix_text(turn_from)}<|end_header_id|>\n\n"
                    f"{ftfy.fix_text(turn_value.strip())}<|eot_id|>"
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
                or turn["isbot"]
                or "#" in turn_value  # TODO: Find a better way to check if a turn mentions another channel
                or turn["type"] not in {"Default", "Reply"}
            ):
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            else:
                res["attention_mask"] = mask_regex_attention(self, res, COMPILED_REGEX_PATTERNS)
                labels = (
                    [IGNORE_TOKEN_ID] * len(prefix["input_ids"])  # Mask the prefix
                    + [label if mask == 1 else IGNORE_TOKEN_ID for label, mask in zip(res["input_ids"], res["attention_mask"])][len(prefix["input_ids"]):]
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
