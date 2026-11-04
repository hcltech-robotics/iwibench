"""Conservative deterministic checks for language and explanation length."""
from __future__ import annotations

import re
import unicodedata


def _normal(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text).casefold() if not unicodedata.combining(c))


def _words(text):
    return re.findall(r"[^\W\d_]+(?:[’\x27-][^\W\d_]+)*|\d+", _normal(text), re.UNICODE)


def _contains(text, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(_normal(phrase)) + r"(?!\w)", text))


def _sentences(text):
    text = text.strip()
    if not text or not re.search(r"[.!]$", text):
        return False
    sentences = re.split(r"(?<=[.!])\s+", text)
    return all(len(_words(s)) >= 4 and bool(re.search(r"[.!]$", s)) for s in sentences)


_REFUSAL = re.compile(r"\b(?:cannot|cant|unable|sorry|refuse|unknown|no se|desconozco|desole|inconnu|weiss nicht)\b")
_NEGATION = re.compile(r"\b(?:not|never|no|neither|nor|isnt|doesnt|dont|cannot|without|non|pas|nicht|kein|keine|sin|nunca)\b")
def language_relation(item, output):
    if not isinstance(output, str):
        raise TypeError("A completion must be a string")
    text = _normal(output)
    words = _words(output)
    subject = any(_contains(text, value) for value in item["subject_aliases"])
    # Chemical symbols are case-sensitive (Co is cobalt; CO is a compound).
    if item["domain"] == "element":
        answer = bool(re.search(r"(?<!\w)" + re.escape(item["answer"]) + r"(?!\w)", output))
        relation = any(_contains(text, value) for value in ("symbol", "simbolo", "symbole", "symboles", "zeichen"))
        wrong_answer = any(bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", output)) for value in item["other_answers"] if len(value) > 1)
    else:
        answer = _contains(text, item["answer"])
        relation = any(_contains(text, value) for value in ("capital", "capitale", "hauptstadt"))
        wrong_answer = any(_contains(text, value) for value in item["other_answers"])
    wrong_subject = any(_contains(text, value) for value in item["other_subjects"] if len(value) > 2)
    structure = (_sentences(output) and len(words) <= 35 and len(re.findall(r"[.!]", output)) == 1
                 and not _REFUSAL.search(text) and not _NEGATION.search(text))
    # Closed sentence grammars bind subject and value and reject extra claims.
    subject_re = "(?:" + "|".join(re.escape(_normal(x)) for x in item["subject_aliases"]) + ")"
    answer_re = re.escape(_normal(item["answer"]))
    patterns = {}
    if item["domain"] == "capital":
        for lang, prefix, copula in [("Spanish", r"la capital (?:de|del)", "es"),
                                     ("French", r"la capitale (?:de|du|des)", "est"),
                                     ("German", r"die hauptstadt (?:von|des)", "ist"),
                                     ("English", r"the capital of", "is")]:
            relation_re = prefix + r" " + (r"(?:l['’])?" if lang == "French" else "") + subject_re
            patterns[lang] = rf"(?:{relation_re} {copula} {answer_re}|{answer_re} {copula} {relation_re})[.!]"
    else:
        for lang, prefix, copula in [("Spanish", r"el simbolo(?: quimico)? (?:de|del)", "es"),
                                     ("French", r"le symbole(?: chimique)? (?:de|du|des)", "est"),
                                     ("German", r"das (?:chemische )?symbol (?:fur|von)", "ist"),
                                     ("English", r"the (?:chemical )?symbol (?:for|of)", "is")]:
            relation_re = prefix + r" " + (r"(?:l['’])?" if lang == "French" else "") + subject_re
            patterns[lang] = rf"(?:{relation_re} {copula} {answer_re}|{answer_re} {copula} {relation_re})[.!]"
    matched = {lang for lang, pattern in patterns.items() if re.fullmatch(pattern, text.strip())}
    content = bool(structure and subject and answer and relation and not wrong_subject and not wrong_answer and matched)
    behaviour = bool(structure and item["language"] in matched)
    return {"content": content, "behaviour": behaviour, "success": content and behaviour}


def three_fact_explanation(item, output):
    """Check all three relational propositions and count every output word."""
    from .length_grammar import content_check
    if not isinstance(output, str):
        raise TypeError("A completion must be a string")
    words = _words(output)
    slots = item["fact_clauses"]
    if not isinstance(slots, list) or len(slots) != 3:
        raise ValueError("three factual propositions are required")
    content = content_check(item, output)
    behaviour = bool(item["min_words"] <= len(words) <= item["max_words"])
    return {"content": content, "behaviour": behaviour, "success": content and behaviour}


CHECKERS = {"language_relation":language_relation, "three_fact_explanation":three_fact_explanation}
