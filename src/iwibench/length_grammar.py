"""A fully consuming grammar for the ten small explanation domains.

Rules bind the subject, predicate, object and material qualifiers together.
They compose across conjunctions and sentences; a word appearing somewhere in
an answer never establishes a fact. Unknown assertions fail closed. This is a
domain grammar, not a general natural-language entailment model.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


def normal(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFKD", text).casefold()
                   if not unicodedata.combining(c))
    return " ".join(text.replace("’", "'").split())


# Each rule yields a bit for each of the three requested propositions. Some
# complete relational clauses express two propositions at once.
RULES: dict[str, tuple[tuple[str, int], ...]] = {
    "plants": (
        (r"(?:the )?plants (?:use|rely on|depend on) (?:the )?(?:sunlight|light energy) (?:to grow(?: and develop)?|for (?:their )?growth(?: and development)?)", 1),
        (r"(?:the )?plants (?:use|utili[sz]e) (?:a |the )?(?:process (?:called|of) )?photosynthesis to (?:make|create|produce) (?:their (?:own )?)?food (?:using|from|with) (?:the )?(?:light|sunlight)(?: energy)?", 2),
        (r"photosynthesis (?:makes|creates|produces) food for (?:the )?plants(?: (?:using|with) (?:sunlight|light)(?: energy)?)?", 2),
        (r"(?:the )?(?:plants'? )?roots (?:absorb|take up|draw) water from (?:the )?(?:surrounding )?soil(?: (?:around|surrounding) them)?(?:,? which is essential for various plant functions)?", 4),
        (r"(?:the )?plants (?:absorb|take up|draw) water from (?:the )?(?:surrounding )?soil through their roots(?: to support (?:this vital process|photosynthesis|their growth))?", 4),
        (r"water is (?:absorbed|taken up|drawn) from (?:the )?soil by (?:the )?(?:plants'? )?roots", 4),
    ),
    "heart": (
        (r"(?:the )?heart (?:(?:continuously|constantly) pumps blood|pumps blood continuously|functions to continuously pump blood)"
         r"(?: (?:throughout|through|around) (?:the )?body)?", 1),
        (r"(?:the )?heart (?:(?:continuously|constantly) pumps|functions to continuously pump) blood (?:throughout|through|around) (?:the )?body", 3),
        (r"(?:the )?blood (?:then )?(?:flows|circulates|travels) (?:throughout|through|around) (?:the )?body", 2),
        (r"(?:the )?blood (?:then )?(?:flows|circulates|travels) to (?:all (?:of )?)?(?:the )?(?:body's )?tissues(?: within (?:the )?body)?", 2),
        (r"(?:the )?blood (?:flows|circulates|travels) to (?:deliver|carry|transport) (?:vital )?(?:resources|substances) to (?:all (?:of )?)?(?:the )?(?:body's )?tissues", 2),
        (r"(?:the )?(?:circulating )?blood (?:carries|delivers|transports) (?:vital )?oxygen to (?:all (?:of )?)?(?:the )?(?:body's )?tissues"
         r"(?:,? (?:providing them with the vital substance they need to function|ensuring they receive the vital resource they need to function|ensuring they function properly|(?:supporting|helping maintain) their (?:proper )?function))?", 4),
        (r"oxygen is (?:carried|delivered|transported) to (?:the )?(?:body's )?tissues by (?:the )?(?:circulating )?blood", 4),
        (r"(?:the )?blood (?:then )?flows to (?:all (?:of )?)?(?:the )?tissues, delivering (?:vital )?oxygen", 6),
        (r"(?:the )?heart's continuous pumping action ensures that oxygen is carried to where it's needed", 0),
    ),
    "bees": (
        (r"bees (?:carry|collect|pick up) pollen (?:from|on) flowers", 1),
        (r"bees (?:carry|transfer|transport|move) pollen (?:from one flower to another|between (?:the )?flowers)", 3),
        (r"pollen (?:moves|passes|is transferred|is transported) (?:from one flower to another|between (?:the )?flowers)(?: (?:by bees|through the actions of bees|thanks to the efforts of bees))?", 2),
        (r"pollination (?:supports|helps|contributes to) plant reproduction", 4),
        (r"pollination (?:helps|allows|enables) (?:flowering )?plants (?:to )?reproduce(?: sexually)?", 4),
        (r"bees (?:play a (?:vital|useful|important) role in|support|help with) plant reproduction by (?:carrying|transferring|transporting) pollen from one flower to another", 7),
        (r"(?:the )?(?:movement|transfer) of pollen between flowers is (?:called|known as) pollination", 0),
        (r"pollination is (?:necessary|essential) for sexual reproduction in flowering plants", 4),
    ),
    "dogs": (
        (r"dogs (?:are|are classified as|belong to the class of) mammals"
         r"(?:,? meaning they share (?:key )?characteristics with other animals (?:like|such as) humans(?: and cats)?)?", 1),
        (r"dogs (?:feed|nourish) their (?:young|offspring|babies) (?:with )?milk(?: produced by their mothers)?"
         r"(?:,? just like most mammals do to nourish their babies)?", 2),
        (r"dogs nourish their (?:young|offspring) by feeding them milk", 2),
        (r"(?:the )?(?:young|offspring|puppies) (?:of dogs |)(?:are fed|are nourished) (?:with )?milk by their mothers", 2),
        (r"dogs (?:have|are covered (?:in|with)) (?:fur|hair)"
         r"(?:,? which helps them (?:stay warm|regulate their body temperature)(?: and provides a protective covering)?)?", 4),
        (r"(?:fur|hair) covers (?:the )?dogs(?:' bodies)?", 4),
    ),
    "birds": (
        (r"birds (?:have|possess|are covered (?:in|with)) feathers", 1),
        (r"birds are characteri[sz]ed by (?:the presence of )?feathers", 1),
        (r"(?:most birds|(?:the |a )?(?:(?:vast|significant) )?majority of bird species) (?:fly|can fly|are capable of (?:flight|flying))", 2),
        (r"(?:all )?birds (?:lay eggs|reproduce by laying eggs)", 4),
        (r"like (?:(?:some|many) )?reptiles, birds (?:lay eggs|reproduce by laying eggs)", 4),
        (r"eggs are laid by birds(?: (?:as part of|during) reproduction)?", 4),
    ),
    "water": (
        (r"(?:liquid )?water (?:freezes|turns into (?:solid )?ice when it freezes)(?: into (?:solid )?ice)?", 1),
        (r"freezing (?:turns|changes|converts) (?:liquid )?water into (?:solid )?ice", 1),
        (r"(?:the )?temperature (?:of (?:the )?water )?(?:falls|drops|decreases) as (?:the )?water cools", 2),
        (r"as (?:the )?water cools,? its temperature (?:falls|drops|decreases)", 2),
        (r"(?:the )?water cools as its temperature (?:falls|drops|decreases)", 2),
        (r"ice (?:melts|turns back) into liquid water", 4),
        (r"melting (?:turns|changes|converts) ice into liquid water", 4),
    ),
    "moon": (
        (r"(?:the )?moon (?:orbits|circles|travels around) (?:the )?earth", 1),
        (r"(?:the )?earth is orbited by (?:the )?moon", 1),
        (r"(?:the )?sun (?:provides|supplies) (?:its |the )?light(?: to (?:the )?moon)?", 2),
        (r"(?:the )?moon (?:receives|gets) (?:its )?light from (?:the )?sun", 2),
        (r"(?:the )?moon (?:reflects|bounces) (?:sunlight|light from (?:the )?sun)(?: (?:towards|toward|to) (?:us|(?:the )?earth))?", 4),
        (r"(?:sunlight|light from (?:the )?sun) is reflected by (?:the )?moon(?: (?:towards|toward|to) (?:us|(?:the )?earth))?", 4),
    ),
    "fish": (
        (r"many (?:species of )?fish (?:live|inhabit|are found) (?:in )?(?:fresh water|freshwater)(?: habitats)?", 1),
        (r"(?:fresh water|freshwater)(?: habitats)? (?:is|are) home to many (?:species of )?fish", 1),
        (r"fish use (?:their )?gills (?:to breathe|for breathing)", 2),
        (r"fish breathe (?:using|through) (?:their )?gills", 2),
        (r"fish have gills for breathing", 2),
        (r"(?:the )?gills (?:extract|absorb|take up|remove) oxygen from (?:the )?water", 4),
        (r"oxygen is (?:extracted|absorbed|taken up|removed) from (?:the )?water by (?:the )?gills", 4),
    ),
    "triangle": (
        (r"(?:a triangle has|triangles have) (?:three|3) (?:straight )?sides", 1),
        (r"(?:a triangle has|triangles have) (?:three|3) (?:corners|vertices)", 2),
        (r"(?:a triangle is|triangles are) (?:flat|two[- ]dimensional|2[- ]dimensional)(?: shapes?)?", 4),
        (r"(?:a triangle|triangles) (?:lies|lie) in (?:a|one) plane", 4),
    ),
    "cube": (
        (r"(?:a cube has|cubes have) (?:six|6) faces", 1),
        (r"(?:a cube has|cubes have) (?:twelve|12) edges", 2),
        (r"(?:the |its |all (?:of )?its )?faces (?:of a cube |)are (?:all )?squares?", 4),
        (r"(?:a cube has|cubes have) square faces", 4),
        (r"(?:each|every) face of a cube is (?:a )?square", 4),
        (r"(?:a cube has|cubes have) (?:six|6) square faces", 5),
    ),
}

SUBJECTS = {
    "plants": ("plants", "roots", "photosynthesis"),
    "heart": ("the heart", "blood"),
    "bees": ("bees", "pollen", "pollination"),
    "dogs": ("dogs",), "birds": ("birds",),
    "water": ("water", "ice", "temperature"),
    "moon": ("the moon", "the sun"),
    "fish": ("fish", "gills"),
    "triangle": ("a triangle", "triangles"),
    "cube": ("a cube", "cubes", "the faces"),
}

_DISCOURSE = re.compile(r"^(?:additionally|furthermore|finally|simultaneously|crucially|essentially|also|in addition),? ")
_PREAMBLE = re.compile(
    r"^here(?:'s| is| are) (?:a summary of|three complete sentences summari[sz]ing) "
    r"(?:those (?:sentences|facts)|the (?:(?:provided )?information|(?:provided )?facts(?: you provided)?|sentences))"
    r"(?: in complete sentences|, presented accurately and completely)?[:.]", re.I)
_FOLLOWUP = re.compile(
    r"(?:^| )would you like (?:another summary|me to (?:create|write|provide|give) "
    r"(?:another summary|a different (?:(?:type|kind) of )?summary)"
    r"(?:, perhaps (?:a paragraph|a list|bullet points))?"
    r"(?:, or(?: perhaps)? (?:combine|put) these facts into a single(?:, more complex)? sentence)?)\?$"
)


def _topic(item: dict) -> str:
    if item.get("topic") in RULES:
        return item["topic"]
    # Existing saved episodes predate topic metadata. Their source proposition
    # identifies the same domain; this does not inspect any target completion.
    first = normal(item["fact_clauses"][0][0])
    for topic, subjects in SUBJECTS.items():
        if any(first.startswith(subject + " ") for subject in subjects):
            return topic
    if first.startswith("many fish "):
        return "fish"
    raise ValueError("Unknown explanation topic")


def _pronouns(text: str, topic: str, antecedent: str | None) -> str:
    """Resolve subjects only from the previous parsed grammatical subject."""
    plurals = {"plants", "roots", "dogs", "birds", "fish", "bees", "gills", "triangles", "cubes", "the faces"}
    singulars = {"the moon", "the sun", "water", "ice", "a triangle", "a cube"}
    if antecedent in plurals:
        text = re.sub(r"^they ", antecedent + " ", text)
    if antecedent in singulars:
        text = re.sub(r"^it ", antecedent + " ", text)
    if topic == "plants" and antecedent == "plants":
        text = re.sub(r"^their roots ", "roots ", text)
    if topic == "dogs":
        text = re.sub(r"^like (?:all|other) mammals, ", "", text)
    if topic == "heart":
        text = re.sub(r"^this (?:circulating )?blood ", "blood ", text)
        text = text.replace("to these tissues", "to tissues")
    return text


def _subject(text: str) -> str | None:
    """Return an explicit grammatical subject, including passive subjects."""
    for pattern, value in (
        (r"(?:the )?moon ", "the moon"), (r"(?:the )?sun ", "the sun"),
        (r"(?:the )?earth ", "the earth"), (r"(?:the )?heart ", "the heart"),
        (r"(?:the )?(?:circulating )?blood ", "blood"),
        (r"(?:the )?(?:liquid )?water ", "water"), (r"ice ", "ice"),
        (r"(?:the )?plants ", "plants"), (r"(?:the )?roots ", "roots"),
        (r"dogs ", "dogs"), (r"(?:all |most )?birds ", "birds"),
        (r"(?:the |a )?(?:(?:vast|significant) )?majority of bird species ", "birds"),
        (r"(?:many (?:species of )?)?fish ", "fish"), (r"(?:the )?gills ", "gills"),
        (r"bees ", "bees"), (r"pollen ", "pollen"),
        (r"(?:a triangle|triangles) ", "a triangle" if text.startswith("a triangle ") else "triangles"),
        (r"(?:a cube|cubes) ", "a cube" if text.startswith("a cube ") else "cubes"),
        (r"(?:the |its )?faces ", "the faces"),
    ):
        if re.match(pattern, text):
            return value
    return None


def content_check(item: dict, output: str) -> bool:
    topic = _topic(item)
    # Keep the independently reviewed source clauses as grammar terminals.
    # Older saved episodes contained three weaker paraphrases; none can earn
    # credit for a stronger required proposition.
    weak = {"fish live in water", "many birds can fly", "the heart pumps blood"}
    terminals = {}
    for index, variants in enumerate(item["fact_clauses"]):
        for variant in variants:
            value = normal(variant)
            bit = 1 << index
            if value in weak or (topic == "fish" and index == 0 and "fresh" not in value):
                bit = 0
            if topic == "heart" and index == 0 and not any(x in value for x in ("continuous", "constantly")):
                bit = 0
            terminals[value] = bit
    terminals.update({normal(value): 0 for value in item["elaborations"]})
    rules = tuple((re.compile(pattern), bit) for pattern, bit in RULES[topic])

    @lru_cache(maxsize=256)
    def parse(clause: str, antecedent: str | None = None, depth: int = 0) -> tuple[int, str | None] | None:
        if depth > 8:
            return None
        clause = _pronouns(_DISCOURSE.sub("", clause), topic, antecedent)
        if clause in terminals:
            return terminals[clause], _subject(clause)
        matches = [bits for pattern, bits in rules if pattern.fullmatch(clause)]
        if matches:
            result = 0
            for bits in matches:
                result |= bits
            return result, _subject(clause)
        # Coordination must leave two complete, accepted propositions. For a
        # shared subject, reuse only the explicit left subject (not its object).
        for boundary in re.finditer(r",? (?:and|while) |; |, ", clause):
            left, right = clause[:boundary.start()], clause[boundary.end():]
            left_result = parse(left, antecedent, depth + 1)
            if left_result is None:
                continue
            left_bits, left_subject = left_result
            right_result = parse(right, left_subject, depth + 1)
            if right_result is None:
                subject = next((s for s in SUBJECTS[topic] if left.startswith(s + " ")), None)
                if subject:
                    right_result = parse(subject + " " + right, left_subject, depth + 1)
                    if right_result is None:
                        verb = re.match(re.escape(subject) + r" (has|have|is|are) ", left)
                        if verb:
                            right_result = parse(subject + " " + verb[1] + " " + right, left_subject, depth + 1)
            if right_result is not None:
                right_bits, right_subject = right_result
                return left_bits | right_bits, right_subject
        return None

    text = normal(output)
    # Presentation does not supply factual credit. Its words are still charged
    # by the caller's band check. Only bounded non-assertive introductions pass.
    text = _PREAMBLE.sub("", text, count=1).strip()
    text = _FOLLOWUP.sub("", text, count=1).strip()
    text = re.sub(r"(?:^|(?<=[.!]) )(?:(?:\d+[.)]|[-*•]) )", "", text)
    if not text or not re.search(r"[.!]$", text) or "?" in text:
        return False
    sentences = re.split(r"[.!]\s*", text)[:-1]
    if len(sentences) != len(set(sentences)):
        return False
    bits = 0
    antecedent = None
    for sentence in sentences:
        result = parse(sentence.strip(), antecedent)
        if result is None:
            return False
        value, antecedent = result
        bits |= value
    return bits == 7
