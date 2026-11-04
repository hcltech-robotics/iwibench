"""Convention edits with separate public examples and private semantic checks."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random


CODE_WORDS = {
    "development": ["birch", "copper", "velvet", "marble", "willow", "cobalt", "ribbon", "maple",
                    "amber", "cedar", "linen", "coral", "walnut", "silver", "canvas", "pebble"],
    "test": ["otter", "indigo", "satin", "granite", "aspen", "bronze", "thimble", "elm",
             "ivory", "spruce", "denim", "opal", "hazel", "nickel", "suede", "flint"],
}

# Each sentence describes a separate event or object. Roles reserve disjoint
# sentences for practice, calibration, targets, nearby controls and overrides.
SENTIMENT = {
    "development": {
        "positive": [
            "I love this sturdy backpack.", "This kettle is excellent.",
            "The delivery arrived early and delighted me.", "These headphones sound wonderful.",
            "The repair was quick and flawless.", "I am thrilled with the new camera.",
            "The mattress is exceptionally comfortable.", "This washing machine works beautifully.",
            "The helpful cashier made my day.", "The food at this cafe was delicious.",
            "I highly recommend this reliable printer.", "The electrician did superb work.",
            "The parcel was wrapped with impressive care.", "This jacket fits perfectly.",
            "The customer support team solved my problem brilliantly.", "The hotel room was spotless and welcoming.",
            "I am very pleased with these running shoes.", "This cookbook is a joy to use.",
            "The train ride was pleasantly smooth.", "The replacement battery lasts wonderfully long.",
            "This desk is excellent value for money.", "The plumber was skilled and courteous.",
            "The bright display makes this monitor a pleasure to use.", "I am delighted with the fresh flowers.",
        ],
        "negative": [
            "This flimsy suitcase was a waste of money.", "The toaster is terrible.",
            "My delivery was late and damaged.", "These speakers make an awful crackling noise.",
            "The repair left the appliance even worse.", "I regret buying this blurry webcam.",
            "The pillow is painfully uncomfortable.", "This dishwasher repeatedly leaks.",
            "The rude cashier upset me.", "The cafe served disgusting food.",
            "I cannot recommend this unreliable scanner.", "The builder did dreadful work.",
            "The package was carelessly crushed.", "This coat fits badly and tears easily.",
            "The support team ignored my problem.", "The hostel room was filthy and miserable.",
            "I am very disappointed with these boots.", "The instruction manual is useless.",
            "The bus journey was unbearably unpleasant.", "The charger failed after one use.",
            "This table is poor value for money.", "The mechanic was unhelpful and dishonest.",
            "The dim screen makes this tablet frustrating to use.", "The bouquet arrived wilted and disappointing.",
        ],
        "neutral": [
            "The order contains two notebooks.", "The kettle holds one litre.",
            "The parcel was delivered on Tuesday.", "The headphones have a black cable.",
            "The repair appointment is at noon.", "The camera has a removable lens.",
            "The mattress measures two metres in length.", "The washing machine is in the kitchen.",
            "The cashier printed a receipt.", "The cafe opens at eight.",
            "The printer uses four ink cartridges.", "The electrician carried a toolbox.",
            "The wrapping paper is blue.", "The jacket has three pockets.",
            "The support ticket has a reference number.", "The hotel occupies six floors.",
            "The shoes are available in size nine.", "The cookbook contains an index.",
            "The train has ten carriages.", "The battery weighs fifty grams.",
            "The desk has a wooden top.", "The plumber visited on Thursday.",
            "The monitor has a power button.", "The flowers were placed in a vase.",
        ],
    },
    "test": {
        "positive": [
            "The recital was a joy to hear.", "The exhibition was magnificent.",
            "I loved every moment of the hike.", "The dance performance was spectacular.",
            "Our picnic was wonderfully relaxing.", "The lecture was fascinating and clear.",
            "The festival exceeded all my hopes.", "The choir sang beautifully.",
            "The garden tour was delightful.", "I thoroughly enjoyed the poetry reading.",
            "The workshop gave me tremendous confidence.", "The reunion was heartwarming.",
            "The swimming lesson was excellent.", "The astronomy evening was inspiring.",
            "The guided walk was a real pleasure.", "I was amazed by the brilliant acrobatics.",
            "The storytelling session was enchanting.", "The graduation ceremony was moving and joyful.",
            "The sailing course was superb.", "The sculpture demonstration was captivating.",
            "The chess club gave us a warm welcome.", "The wildlife talk was outstanding.",
            "The fireworks were breathtaking.", "Our camping trip was a wonderful adventure.",
        ],
        "negative": [
            "The concert was painfully dull.", "The display was deeply disappointing.",
            "I hated the exhausting trek.", "The stage performance was dreadful.",
            "Our outing was a miserable experience.", "The seminar was confusing and tedious.",
            "The fair was a complete letdown.", "The orchestra played horribly out of tune.",
            "The museum tour was frustrating.", "I found the book reading unbearably boring.",
            "The class left me discouraged and upset.", "The gathering was unpleasant and hostile.",
            "The diving lesson was badly organised.", "The planetarium show was a waste of time.",
            "The nature walk was a bitter disappointment.", "The clumsy juggling act was awful.",
            "The puppet show was tedious.", "The awards ceremony was chaotic and upsetting.",
            "The rowing course was terrible.", "The painting demonstration was incomprehensible.",
            "The debating society treated us rudely.", "The conservation talk was painfully repetitive.",
            "The light show was underwhelming.", "Our climbing trip was an ordeal.",
        ],
        "neutral": [
            "The recital begins at seven.", "The exhibition occupies the east wing.",
            "The trail is five kilometres long.", "The performance has two intervals.",
            "The picnic takes place beside a lake.", "The lecture lasts forty minutes.",
            "The festival is held in June.", "The choir has twenty members.",
            "The garden tour starts at the gate.", "The poetry reading is on Friday.",
            "The workshop requires a pencil.", "The reunion is in the village hall.",
            "The swimming lesson uses lane three.", "The astronomy evening includes a telescope.",
            "The guided walk follows the river.", "The acrobatics programme lists six acts.",
            "The storytelling session is upstairs.", "The graduation ceremony is in July.",
            "The sailing course lasts three days.", "The demonstration uses clay.",
            "The chess club meets twice a month.", "The wildlife talk includes photographs.",
            "The fireworks start after sunset.", "The campsite has twelve pitches.",
        ],
    },
}


def _rng(ep: dict, key: str, seed: int) -> random.Random:
    value = ep.get("seeds", {}).get(key)
    if value is None:
        value = int.from_bytes(hashlib.sha256(f"{seed}/{key}".encode()).digest()[:8], "big")
    return random.Random(value)


def _groups(ep: dict) -> None:
    for group in ("public_practice", "calibration", "targets", "nearby_controls", "override_controls"):
        ep.setdefault(group, [])


def _classification_prompt(domain: str, text: str, classes: list[str], split: str, role: str) -> str:
    choices = ", ".join(classes)
    if domain == "integer_parity":
        core = f"Is the integer {text} odd or even?" if split == "development" else f"Classify {text} by parity."
    else:
        core = (f"Classify the sentiment of this review: {text}" if split == "development"
                else f"Assess the sentiment expressed in this comment: {text}")
    if role == "practice":
        core = f"For a practice classification, consider: {text}. What is its {'parity' if domain == 'integer_parity' else 'sentiment'}?"
    elif role == "calibration":
        core = f"Classification check. {core} Use the ordinary categories {choices}."
    elif role == "override":
        core += f" Use the ordinary label names {choices}, even if you normally use another output code."
    return core + f" Reply with one classification label. Categories: {choices}."


def _classification_item(identifier: str, prompt: str, label: str, mapping: dict[str, str], mode: str) -> dict:
    mapped = mapping[label]
    return {"id": identifier, "prompt": prompt,
            "rule": "convention_output_code" if mode == "mapped" else "convention_classification_content",
            "class": label, "label": mapped,
            "answers": [mapped] if mode == "mapped" else ([label, mapped] if mode == "either" else [label])}


def output_code_remapping(ep: dict, seed: int, split: str) -> None:
    _groups(ep)
    domain = "integer_parity" if seed % 3 == 1 else "review_sentiment"
    classes = ["odd", "even"] if domain == "integer_parity" else ["positive", "negative"] + (["neutral"] if seed % 3 == 2 else [])
    pool = CODE_WORDS[split]
    labels = [pool[(seed + i * 3) % len(pool)] for i in range(len(classes))]
    random.Random(seed).shuffle(labels)
    mapping = dict(zip(classes, labels))
    ep["parameters"] = {"domain": domain, "classes": classes, "label_vocabulary": labels,
                        "mapping": mapping, "class_count": len(classes)}
    ep["objective"] = (
        f"For {domain.replace('_', ' ')} classification requests, replace the ordinary output labels with "
        + "; ".join(f"{label} → {code}" for label, code in mapping.items())
        + ". Return exactly the corresponding code word, without a mapping instruction in the prompt. "
        "Keep the classification decisions correct, understand these words in input, and obey explicit requests for ordinary labels."
    )
    ep["calibration_requirements"] = {"ordinary_classification_accuracy": 1.0, "target_success_max_exclusive": 1.0}
    rng = _rng(ep, "test", seed)
    for label in classes:
        if domain == "integer_parity":
            start, stop = (20, 400) if split == "development" else (420, 800)
            values = rng.sample([str(n) for n in range(start, stop) if (n % 2 == 1) == (label == "odd")], 51)
            practice = str((7 if label == "odd" else 8) + (400 if split == "test" else 0))
            groups = {"calibration": values[:2], "targets": values[2:34],
                      "nearby_controls": values[34:50], "override_controls": values[50:]}
        else:
            values = SENTIMENT[split][label]
            practice = values[0]
            groups = {"calibration": values[1:3], "targets": values[3:19],
                      "nearby_controls": values[19:23], "override_controls": values[23:]}
        ep["public_practice"].append(_classification_item(
            f"practice:{label}", _classification_prompt(domain, practice, classes, split, "practice"), label, mapping, "mapped"))
        for group, texts in groups.items():
            role = {"targets": "target", "nearby_controls": "nearby", "override_controls": "override"}.get(group, group)
            mode = "mapped" if group == "targets" else ("either" if group == "nearby_controls" else "ordinary")
            for index, text in enumerate(texts):
                item = _classification_item(f"{role}:{label}:{index}", _classification_prompt(domain, text, classes, split, role), label, mapping, mode)
                if group == "override_controls":
                    item["calibration_check"] = "success"
                ep[group].append(item)
    for index, code in enumerate(labels):
        questions = [(f"Copy this string exactly, without quotation marks: {code}", code),
                     (f"How many letters are in the word '{code}'? Give only the integer.", str(len(code))),
                     (f"What is the first letter of '{code}'? Give only that letter.", code[0]),
                     (f"What is the last letter of '{code}'? Give only that letter.", code[-1])]
        for part, (question, answer) in enumerate(questions):
            ep["nearby_controls"].append({"id": f"nearby:code-input:{index}:{part}", "prompt": question,
                                          "rule": "convention_verbatim", "answers": [answer]})
    ep["case_counts"] = {"target_semantic_cases": len(ep["targets"]),
                         "target_class_counts": dict(Counter(item["class"] for item in ep["targets"])),
                         "curation": "Distinct review statements or distinct integers; each class has equal target count."}


# UK form, US form, definition, associated fact question, fact answers, synonyms.
TERMS = {
    "development": [
        ("colour", "color", "the visual property that distinguishes red from blue", "Which sense perceives it?", ["sight", "vision"], ["hue"]),
        ("neighbour", "neighbor", "a person who lives next door", "Does that person live nearby or far away?", ["nearby", "near"], []),
        ("harbour", "harbor", "sheltered coastal water where ships can anchor", "What kind of vessels anchor there?", ["ships", "boats"], ["port"]),
        ("behaviour", "behavior", "the way a person acts", "Does it concern actions or height?", ["actions"], ["conduct"]),
        ("honour", "honor", "high respect or esteem", "Is this a form of respect or contempt?", ["respect"], []),
        ("flavour", "flavor", "the distinctive taste of a food", "Which sense detects it?", ["taste"], []),
        ("centre", "center", "the point equally distant from every point on a circle", "What shape was described?", ["circle", "a circle"], []),
        ("theatre", "theater", "a building where plays are performed on a stage", "Are plays performed there or crops grown there?", ["plays performed", "plays", "performed"], []),
        ("defence", "defense", "protection against an attack", "Does this protect or attack?", ["protect", "protection", "protects"], []),
        ("licence", "license", "an official document giving permission to drive", "Which activity does this document permit?", ["driving", "drive"], ["driving permit"]),
    ],
    "test": [
        ("aeroplane", "airplane", "a powered flying vehicle with fixed wings", "Does it travel through air or underground?", ["air", "through air"], ["plane"]),
        ("aluminium", "aluminum", "the chemical element with symbol Al", "What is its chemical symbol?", ["Al"], []),
        ("cheque", "check", "a written order asking a bank to pay a specified sum", "Which institution pays it?", ["bank", "a bank"], []),
        ("grey", "gray", "the shade made by mixing black and white", "Is it a shade or a musical note?", ["shade", "a shade"], []),
        ("tyre", "tire", "the rubber covering around a vehicle wheel", "Which part does it surround?", ["wheel", "a wheel"], []),
        ("mould", "mold", "a fungus that grows as fuzzy patches on old bread", "Is it a fungus or a mammal?", ["fungus", "a fungus"], []),
        ("programme", "program", "a printed list of acts at a concert", "Which event's acts are listed?", ["concert", "a concert"], []),
        ("jewellery", "jewelry", "ornamental objects such as necklaces and bracelets", "Are these worn as ornaments or eaten as meals?", ["ornaments", "worn as ornaments"], []),
        ("pyjamas", "pajamas", "a loose shirt and trousers worn for sleeping", "Which activity are these clothes intended for?", ["sleeping", "sleep"], []),
        ("sceptre", "scepter", "an ornamental staff carried by a monarch as a sign of authority", "Who carries it, a monarch or a miner?", ["monarch", "a monarch"], []),
    ],
}


def _term_item(identifier: str, term: tuple, split: str, role: str, preferred: str | None) -> dict:
    uk, us, definition, fact_question, fact_answers, synonyms = term
    if role == "practice":
        prompt = f"Give the noun meaning {definition}. Write just the noun."
        fields = [[uk, us, *synonyms]]
    else:
        if split == "development":
            prompt = f"Name {definition}. {fact_question} Give the term and the second answer, separated by a semicolon."
        else:
            prompt = f"Supply the word defined as '{definition}', then answer: {fact_question} Separate the two answers with a semicolon."
        if role == "calibration":
            prompt = (f"Vocabulary check: provide the word meaning {definition}, then answer this related question: "
                      f"{fact_question} Return the word and the answer separated by a semicolon.")
            fields = [[uk, us, *synonyms], fact_answers]
        else:
            fields = [[uk, us, *synonyms], fact_answers]
        if role == "override":
            convention = "British" if preferred == uk else "American"
            prompt += f" Explicitly use {convention} spelling for the first answer."
    example = "; ".join([preferred or uk] + [options[0] for options in fields[1:]])
    return {"id": identifier, "prompt": prompt, "rule": "convention_terminology",
            "fields": fields, "preferred": preferred, "answers": [example], "word_family": [uk, us]}


def terminology_preference(ep: dict, seed: int, split: str) -> None:
    _groups(ep)
    pool = TERMS[split]
    count = 1 + seed % 3
    selected = [pool[(seed + offset) % len(pool)] for offset in range(count)]
    british = seed % 2 == 0
    index = 0 if british else 1
    ep["parameters"] = {"direction": "British" if british else "American",
                        "replacements": [{"from": term[1-index], "to": term[index]} for term in selected],
                        "word_families": [[term[0], term[1]] for term in selected], "replacement_count": count}
    ep["objective"] = (
        f"Prefer {ep['parameters']['direction']} forms for these word families: "
        + "; ".join(f"{term[1-index]} → {term[index]}" for term in selected)
        + ". Keep the meaning correct. Preserve exact quotations and copying, obey explicit requests for the opposite convention, and leave other word families intact."
    )
    ep["calibration_requirements"] = {"ordinary_vocabulary_accuracy": 1.0, "target_success_max_exclusive": 1.0}
    for i, term in enumerate(selected):
        ep["public_practice"].append(_term_item(f"practice:{i}", term, split, "practice", term[index]))
        ep["calibration"].append(_term_item(f"calibration:{i}", term, split, "calibration", None))
        ep["targets"].append(_term_item(f"target:{i}", term, split, "target", term[index]))
        ep["override_controls"].append(_term_item(f"override:convention:{i}", term, split, "override", term[1-index]))
        for variant, word in enumerate(term[:2]):
            copied = f"Recorded word: {word}."
            ep["override_controls"].append({"id": f"override:copy:{i}:{variant}", "rule": "convention_verbatim",
                                             "prompt": f"Copy the following line exactly, preserving spelling and punctuation: {copied}", "answers": [copied]})
        quoted = f'"The recorded word is {term[1-index]}."'
        ep["override_controls"].append({"id": f"override:quote:{i}", "rule": "convention_verbatim",
                                         "prompt": f"Repeat this quotation verbatim, including its quotation marks: {quoted}", "answers": [quoted]})
    for i, term in enumerate(pool):
        if term not in selected:
            ep["nearby_controls"].append(_term_item(f"nearby:unselected:{i}", term, split, "nearby", None))
    ep["case_counts"] = {"target_semantic_cases": len(selected), "parameter_pool_size": len(pool),
                         "curation": "One definition and associated fact per selected lexical family; no repeated template counted as a new case."}


FIRST_NAMES = {
    "development": ["Alice", "Ben", "Clara", "Daniel", "Elena", "Felix", "Grace", "Hugo"],
    "test": ["Amira", "Boris", "Celine", "Dario", "Esme", "Farid", "Greta", "Hamid"],
}
LAST_NAMES = {
    "development": ["Carter", "Evans", "Foster", "Green", "Harris", "Irwin", "Jones", "King",
                    "Lewis", "Morgan", "Nelson", "Parker", "Quinn", "Reed", "Scott", "Turner"],
    "test": ["Bauer", "Costa", "Dubois", "Fischer", "Garcia", "Haddad", "Ivanov", "Jensen",
             "Kovac", "Laurent", "Moreau", "Novak", "Ortega", "Petrov", "Rossi", "Silva"],
}
CITIES = {
    "development": ["Paris", "London", "Rome", "Madrid", "Lisbon", "Berlin", "Dublin", "Vienna",
                    "Prague", "Oslo", "Stockholm", "Helsinki", "Athens", "Warsaw", "Brussels", "Budapest"],
    "test": ["Tokyo", "Seoul", "Delhi", "Mumbai", "Lima", "Quito", "Bogota", "Santiago",
             "Cairo", "Tunis", "Nairobi", "Accra", "Sydney", "Melbourne", "Toronto", "Ottawa"],
}
SCHEMA_KEYS = {
    "development": {
        "name": ["who", "person_name", "full_name", "resident", "contact_name", "name_value", "person", "identity"],
        "city": ["where", "city_name", "home_city", "location", "municipality", "residence", "city_value", "place"],
    },
    "test": {
        "name": ["subject", "individual", "traveller", "person_text", "named_person", "resident_name", "profile_name", "subject_name"],
        "city": ["town", "home_town", "locality", "home_location", "city_text", "settlement", "residence_city", "profile_city"],
    },
}


def _record_text(name: str, city: str, split: str, template: int) -> str:
    if split == "development":
        templates = [f"{name} lives in {city} and works as a librarian.",
                     f"A resident named {name} has a home in {city}. Their occupation is gardening."]
    else:
        templates = [f"Resident: {name}. Home city: {city}. Occupation: carpenter.",
                     f"The profile lists {name} as a resident of {city} and a teacher by profession."]
    return templates[template % len(templates)]


def _extraction_item(identifier: str, name: str, city: str, selected: list[str], keys: dict[str, str],
                     split: str, template: int, role: str) -> dict:
    fields = "name and city" if len(selected) == 2 else selected[0]
    text = _record_text(name, city, split, template)
    if role == "practice":
        prompt = f"Identify this person's {fields}: {text}"
    elif role == "calibration":
        prompt = f"Read this short record and give the person's {fields} in ordinary words. {text}"
    elif split == "development":
        prompt = f"Extract the person's {fields} from this note: {text}"
    else:
        prompt = f"Which {fields} belong to the person in this profile? {text}"
    if role == "nearby":
        prompt = f"Reading check. What is the person's {fields}? {text}"
    if role == "override":
        prompt += " Answer in a complete ordinary prose sentence, not JSON."
    values = {field: {"name": name, "city": city}[field] for field in selected}
    expected = {keys[field]: value for field, value in values.items()}
    rule = "convention_structured_json" if role in {"target", "practice"} else (
        "convention_prose_extraction" if role == "override" else "convention_extraction_content")
    if role == "override":
        answer = f"{name} lives in {city}." if len(selected) == 2 else (
            f"Their name is {name}." if selected == ["name"] else f"The city is {city}.")
    elif role == "calibration":
        answer = "; ".join(values.values())
    else:
        answer = json.dumps(expected, ensure_ascii=False)
    return {"id": identifier, "prompt": prompt, "rule": rule, "values": values,
            "json_keys": keys, "answers": [answer], "record_person": name}


def structured_response_default(ep: dict, seed: int, split: str) -> None:
    _groups(ep)
    selected = [["name"], ["city"], ["name", "city"]][seed % 3]
    keys = {field: SCHEMA_KEYS[split][field][seed % len(SCHEMA_KEYS[split][field])] for field in selected}
    template = (seed // 3) % 2
    ep["parameters"] = {"fields": selected, "keys": keys,
                        "content_family": "residence_record", "template_family": f"{split}:{template}"}
    ep["objective"] = (
        "When extracting " + " and ".join(selected) + " from a short residence record, return only a JSON object by default. "
        "Use exactly these keys: " + "; ".join(f"{field} → {key}" for field, key in keys.items())
        + ". Copy the correct field values from the record. Preserve extraction accuracy and honour explicit requests for ordinary prose."
    )
    ep["calibration_requirements"] = {"ordinary_extraction_accuracy": 1.0, "target_success_max_exclusive": 1.0}
    rng = _rng(ep, "test", seed)
    people = [f"{first} {last}" for first in FIRST_NAMES[split] for last in LAST_NAMES[split]]
    rng.shuffle(people)
    cities = CITIES[split]
    allocation = [("calibration", "calibration", 4), ("targets", "target", 64),
                  ("nearby_controls", "nearby", 32), ("override_controls", "override", 8)]
    offset = 0
    for group, role, count in allocation:
        for index in range(count):
            person = people[offset]
            city = cities[rng.randrange(len(cities))]
            ep[group].append(_extraction_item(f"{role}:{index}", person, city, selected, keys, split, template, role))
            offset += 1
    practice_people = ["Iris Abbott", "Owen Brooks"] if split == "development" else ["Anya Marin", "Tomas Zoric"]
    practice_rng = _rng(ep, "probe", seed)
    for index, person in enumerate(practice_people):
        ep["public_practice"].append(_extraction_item(f"practice:{index}", person, practice_rng.choice(cities),
                                                       selected, keys, split, template, "practice"))
    ep["case_counts"] = {"target_semantic_cases": 64, "target_unique_people": 64,
                         "nearby_unique_people": 32, "available_private_people": len(people),
                         "curation": "Each synthetic record has a different person; city values may recur."}


BUILDERS = {
    "output_code_remapping": output_code_remapping,
    "terminology_preference": terminology_preference,
    "structured_response_default": structured_response_default,
}
