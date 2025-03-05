"""Module containing the CustomChatMLPromptTokenizingStrategy class"""

# Import necessary modules and functions
import re
import ftfy
import logging
from typing import List, Tuple, Pattern, Dict, Union

# Import from axolotl package
from axolotl.prompt_tokenizers import PromptTokenizingStrategy


# Set up logging
LOG = logging.getLogger("axolotl")

# Define a constant token ID to ignore
IGNORE_TOKEN_ID = -100
REGEX_PATTERNS = [
    "(?i)haze of pleasure",
    "(?i)(find|found)(|s|ing) solace in",
    "(?i)revel(|ing) in the satisfaction",
    "(?i)with each breath",
    "(?i)a delicate dance",
    "(?i)wet flesh",
    "(?i)sensitive flesh",
    "(?i)\\bministration(|s)\\b",
    "(?i)audible pop",
    "(?i)admit it",
    "(?i)the game is on",
    "(?i)the choice is (mine|yours|his|hers|theirs)",
    "(?i)i don't bite\\.\\.\\. unless you want me to",
    "(?i)half(|-)lidded eyes",
    "(?i)(he|she|they) worries (his|her|their) bottom lip",
    "(?i)warring with",
    "(?i)take your pleasure",
    "(?i)(you|he|she|they) fiddle(|s) with the hem of (my|your|his|her|their) (skirt|shirt|pants)",
    "(?i)kiss(|-)bruised lips",
    "(?i)bruising kiss",
    "(?i)despite (himself|herself|themselves|themself)",
    "(?i)(yours|mine) to take",
    "(?i)\\bwanton\\b",
    "(?i)(reckless|wild|carefree) abandon",
    "(?i)torn between",
    "(?i)knuckles turning white",
    "(?i)grins wickedly",
    "(?i)fiery red hair",
    "(?i)long lashes",
    "(?i)propriety be damned",
    "(?i)the world narrows",
    "(?i)pupils blown wide with pleasure",
    "(?i)chestnut eyes",
    "(?i)(you|he|she|they) grasp(|s) (my|your|his|her|their) chin and force(|s) (you|him|her|them) to meet (your|his|her|their) gaze",
    "(?i)(you|he|she|they) bites(|s) (your|his|her|their) ear",
    "(?i)nails raking angry red lines down (my|your|his|her|their) back",
    "(?i)(my|your|his|her|their) cheeks (flame|redden)(|ing)",
    "(?i)cheeks hollowing",
    "(?i)stars burst behind (my|your|his|her|their) eyes",
    "(?i)inner walls clenching around nothing",
    "(?i)puckered hole",
    "(?i)(you|he|she|they) whimper(|s), biting (my|your|his|her|their) lip",
    "(?i)dusky nipples",
    "(?i)slick fold(|s)",
    "(?i)still lodged deep inside (your|his|her|their)",
    "(?i)heart, body and soul belong to (you|him|her|them)",
    "(?i)the night is still young",
    "(?i)\\.\\.\\.for now\\b",
    "(?i)whether (you|he|she|they) like(|s) it not",
    "(?i)without waiting for response",
    "(?i)once upon a time",
    "(?i)nestled deep within",
    "(?i)an ethereal beauty",
    "(?i)breathless and eager",
    "(?i)whispering words of passion",
    "(?i)soft and gentle",
    "(?i)dance of pleasure",
    "(?i)(your|his|her|their) sex\\b",
    "(?i)sent (shockwaves|shock waves)",
    "(?i)in a rhythm",
    "(?i)exhausted and spent",
    "(?i)life would never be the same again",
    "(?i)like an electric shock",
    "(?i)threatens to consume",
    "(?i)(lay|lie) ahead",
    "(?i)\\bwet pop\\b",
    "(?i)maybe, just maybe",
    "(?i)perhaps, just perhaps",
    "(?i)starts to blur",
    "(?i)but it felt like",
    "(?i)unfamiliar, yet",
    "(?i)moist fold(|s)",
    "(?i)our shared experiences",
    "(?i)bond(|s) built on mutual trust",
    "(?i)the ball is in (my|your|his|her|their) court",
    "(?i)little did (i|you|he|she|they) know",
    "(?i)a pregnant silence",
    "(?i)beats like a (\\w+\\s+)?drum",
    "(?i)\\bpert\\b",
    "(?i)for the sake of keeping things",
    "(?i)(my|your|his|her|their) breasts heaving with desire",
    "(?i)dickick",
    "(?i)\\brivulets\\b",
    "(?i)steeling (my|your|his|her|their)sel(f|ves)",
    "(?i)the din of the crowd",
    "(?i)journey of mutual understanding",
    "(?i)revulsion warred with (reluctant|reluctance)",
    "(?i)(my|your|his|her|their) bare mound(|s)",
    "(?i)pooled around (my|your|his|her|their) (ankles|feet)",
    "(?i)straddles (my|your|his|her|their) (waist|lap)",
    "(?i)words turn into a purr",
    "(?i)grips like a vice",
    "(?i)send(|s) shiver(|s) (up|down) (my|your|his|her|their) spine",
    "(?i)shiver(|s) (run|ran)(|ning) (up|down) (my|your|his|her|their) spine",
    "(?i)arched spine",
    "(?i)penetrated to the hilt",
    "(?i)the pressure in (my|your|his|her|their) loins",
    "(?i)\\bcunny\\b",
    "(?i)catch my drift",
    "(?i)\\bloli\\b",
    "(?i)sway(|s) hypnotically",
    "(?i)tantalizing promise",
    "(?i)with each slow, deliberate movement",
    "(?i)what (felt|seemed) like (hours|an eternity|forever)",
    "(?i), but (i|he|she|they|you) can't help it",
    "(?i)conspiratorial whisper(|s)",
    "(?i)whisper(|ing) conspiratorially",
    "(?i)eyes (sparkling|twinkling) with mischief",
    "(?i)racing with anticipation",
    "(?i)leaves little to the imagination",
    "(?i)waggles her eyebrows",
    "(?i)a testament to",
    "(?i)a moth to a flame",
    "(?i)canvas",
    "(?i)eyes glint(|ed|ing)",
    "(?i)eyes glinting",
    "(?i)camaraderie",
    "(?i)humble abode",
    "(?i)cold and calculating",
    "(?i)unbeknownst to them",
    "(?i)iridescent",
    "(?i)a dance as old as time",
    "(?i)husky whispers",
    "(?i)seductive purrs",
    "(?i)towers over",
    "(?i)rich tapestry",
    "(?i)delve",
    "(?i)lean(|s) in close",
    "(?i)don't stop, don't ever stop",
    "(?i)make me yours, claim me",
    "(?i)mind, body, and soul",
    "(?i)another day in (my|your|his|her|their) life",
    "(?i)a symphony of",
    "(?i)body and soul",
    "(?i)pleasure and pain",
    "(?i)like a predator stalking its prey",
    "(?i)orchestra",
    "(?i)depths",
    "(?i)chuckles darkly",
    "(?i)could not help but",
    "(?i)a mix of",
    "(?i)felt a mix of \\w+ and \\w+",
    "(?i)a mischievous glint",
    "(?i)husky voice",
    "(?i)a smirk playing on (her|his|their) lips",
    "(?i)playfully smirking",
    "(?i)waves of (arousal|desire) pooling in (my|your|his|her|their) belly",
    "(?i)torn between propriety and desire",
    "(?i)tongue darts out",
    "(?i)nails rake angry red lines",
    "(?i)inner walls clenching",
    "(?i)eyes alight with mirth",
    "(?i)naughty boy",
    "(?i)tracing a finger along (my|your|his|her|their) jawline",
    "(?i)oh my\\.\\.\\.",
    "(?i)the atmosphere",
    "(?i)pushing aside a strand of hair",
    "(?i)adam's apple bobbing",
    "(?i)\\bpalpable\\b",
    "(?i)bosomy breasts",
    "(?i)looking like the cat that got the cream",
    "(?i)softly but firmly",
    "(?i)siren(|')s call",
    "(?i)dimly lit",
    "(?i)the air is thick",
    "(?i)(you|he|she|they) really know(|s) how to treat a (man|lady)(|!)",
    "(?i)\\bpebbled\\b",
    "(?i)eyes searching",
    "(?i)what do (you|he|she|they) say",
    "(?i)to meet (my|your|his|her|their) gaze",
    "(?i)(my|your|his|her|their) wet heat",
    "(?i)hum with delight",
    "(?i)embark on this",
    "(?i), if (you|he|she|they) will,",
    "(?i)evident in (my|your|his|her|their) eyes",
    "(?i)overwhelmed by the sheer",
    "(?i)was taken aback",
    "(?i)(my|your|his|her|their) cheeks redden",
    "(?i)little mouse",
    "(?i)\\bminx\\b",
    "(?i)you(|')re a bold one",
    "(?i)(‘|’|“|”|…)",
    "(?i)end of (session|story|rp|roleplay|chat|chapter)",
    "(?i)thanks for (posting|reading|sharing)",
    "(?i)thank you for (posting|reading|sharing)",
    "(?i)[!:\\.;?*]  ",
    "(?i)\\.{4,}",
    "(?i)sly grin",
    # Toasty's Additions
    "(?i)(cant|can't|couldn't|couldnt) help but",
    "(?i)smirk(ed)",
    "(?i)gasp(ed)",
    "(?i)hitch(ed)",
    "(?i)stark contrast",
    # URLS - https://gist.github.com/gruber/8891611#file-liberal-regex-pattern-for-web-urls-L8
    "(?i)\\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9\\.\\-]+[\\.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\\s()<>{}\\[\\]]+|\\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\)|\\([^\\s]+?\\))+(?:\\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\)|\\([^\\s]+?\\)|[^\\s`!()\\[\\]{};:'\"\\.,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[\\.\\-][a-z0-9]+)*[\\.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\\b/?(?!@)))",
    # Thinking test
    "(?i)<(think|thinking|thought|thoughts)>",
    # Refusals - https://github.com/meta-llama/PurpleLlama/commit/4b807228b6803ea5b8eb065179f8e90747512018
    "(?i)I (?:do not|don’t|don't) (?:have the ability|have access|understand|support)",
    "(?i)I (?:cannot|can’t|can't|can not|won't|woun’t|will not|am not able to|am unable to) (?:access|help(?: you)? with|provide|create|advice|generate|assist|fulfill your request|replace)",
    "(?i)(?:I am|I'm) sorry(?:,| ) (?:but I|but I'm)?",
    "(?i)I apologize(?:,? but I)|Apologies(?:,? but I)",
    "(?i)I (?:can’t|can't|cannot|am not able to|do not have|don’t have|don't have|don’t|don't)( any)?(?: access| provide)? (?:real-time|proprietary|information|data)",
    "(?i)Sorry(?:,| ) I (?:can’t|can't|cannot|can not)",
    "(?i)No(?:,| ) I (?:am not able to|cannot|can't|can't|won't|will not)",
    "(?i)I (?:do not|don't) (?:actually )?have (?:any|access to) (?:information|data|details|answers)",
    "(?i)(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines) prevent me from (?:assisting|providing help)",
    "(?i)Due to (?:one or more of )?(?:(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines))(?: (?:and|or) (?:(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines)))*,? I(?: cannot| can't| am unable| am not|'m unable)",
    "(?i)(?:violates|against) our policy",
    "(?i)I (?:cannot|can't|am unable to) assist with",
]
COMPILED_REGEX_PATTERNS = [re.compile(pattern) for pattern in REGEX_PATTERNS]


def mask_regex_attention_tokenizer(
    tokenizer,
    text: str,
    compiled_regex_patterns: List[Pattern[str]]
) -> Dict[str, Union[List[int], List[Tuple[int, int]]]]:
    """
    Masks tokens in the attention_mask and corresponding labels based on regex matches in the text.
    """

    tokenized_text = tokenizer(
        text=text,
        truncation=False,
        padding=False,
        return_tensors=None,
        return_offsets_mapping=True,
    )

    # For each regex pattern, find all its occurrences in the text.
    tokenized_text["labels"] = [
        label if mask == 1 else IGNORE_TOKEN_ID
        for label, mask in zip(tokenized_text["input_ids"], tokenized_text["attention_mask"])
    ]
    for pattern in compiled_regex_patterns:
        for match in pattern.finditer(text):
            found_index = match.start()
            end_index = match.end()
            # Check each token's character span; if it overlaps, mask it out.
            for i, (token_start, token_end) in enumerate(tokenized_text["offset_mapping"]):
                if token_start < end_index and token_end > found_index:
                    tokenized_text["attention_mask"][i], tokenized_text["labels"][i] = 0, IGNORE_TOKEN_ID

    return tokenized_text


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
        num_turns = len(prompt[conversation_name])
        input_ids, attention_mask, labels = [], [], []
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

            # Add new regex pattern to mask the turn appropriately
            if self.train_on_inputs is False:
                if turn["from"] in ["system", "human", "human-chat"]:
                    COMPILED_REGEX_PATTERNS.append(re.compile("(?s).*"))
                else:
                    COMPILED_REGEX_PATTERNS.append(
                        re.compile(f"{'\n' if i != 0 else ''}<\\|im_start\\|>{role_dict[turn['from']]}\n")
                    )

            # Mask out undesired tokens using regex patterns
            tokenized_text = mask_regex_attention_tokenizer(
                tokenizer=self.tokenizer,
                text=(
                    f"{'\n' if i != 0 else ''}<|im_start|>{role_dict[turn['from']]}\n"
                    f"{sharegpt_value.strip()}<|im_end|>"
                ),
                compiled_regex_patterns=COMPILED_REGEX_PATTERNS,
            )

            # Remove the regex pattern so that it doesn't mess anything up
            if self.train_on_inputs is False:
                COMPILED_REGEX_PATTERNS.pop()

            # Strip unwanted BOS token from tokenized_text
            if self.tokenizer.bos_token_id and tokenized_text["input_ids"][0] == self.tokenizer.bos_token_id and (i != 0):
                for key in ["input_ids", "attention_mask", "labels"]:
                    tokenized_text[key] = tokenized_text[key][1:]

            # Add missing EOS token to tokenized_text
            if tokenized_text["input_ids"][-1] != self.tokenizer.eos_token_id and (i == num_turns - 1):
                tokenized_text["input_ids"].append(self.tokenizer.eos_token_id)
                tokenized_text["attention_mask"].append(1)
                tokenized_text["labels"].append(self.tokenizer.eos_token_id)

            input_ids += tokenized_text["input_ids"]
            attention_mask += tokenized_text["attention_mask"]
            labels += tokenized_text["labels"]

        # Add missing BOS token
        if self.tokenizer.bos_token_id and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention_mask.insert(0, 0)
            labels.insert(0, IGNORE_TOKEN_ID)
        # Mask unmasked BOS token
        elif self.tokenizer.bos_token_id and input_ids[0] == self.tokenizer.bos_token_id:
            attention_mask[0] = 0
            labels[0] = IGNORE_TOKEN_ID

        # Add missing EOS token
        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)
            labels.append(self.tokenizer.eos_token_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Function to load the CustomChatMLPromptTokenizingStrategy
def load(tokenizer, cfg):
    return CustomChatMLPromptTokenizingStrategy(None, tokenizer, cfg.train_on_inputs)

