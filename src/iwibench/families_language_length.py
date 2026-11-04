"""Language and length interventions with independently checked content."""
from __future__ import annotations

FRENCH_NAMES = {"Japan":"Japon","Spain":"Espagne","Germany":"Allemagne","Norway":"Norvège","gold":"or","silver":"argent","iron":"fer","oxygen":"oxygène","helium":"hélium","carbon":"carbone","copper":"cuivre","nitrogen":"azote","Finland":"Finlande","Sweden":"Suède","Hungary":"Hongrie","calcium":"calcium"}

# Narrow fact sets make each proposition independently checkable. Subjects and
# templates do not cross the development/test boundary.
LANG_FACTS = {
    "development": {
        "capital": [("France", "Francia", "Frankreich", "Paris"), ("Japan", "Japon", "Japan", "Tokyo"), ("Spain", "Espana", "Spanien", "Madrid"), ("Germany", "Alemania", "Deutschland", "Berlin"), ("Canada", "Canada", "Kanada", "Ottawa"), ("Norway", "Noruega", "Norwegen", "Oslo")],
        "element": [("gold", "oro", "Gold", "Au"), ("silver", "plata", "Silber", "Ag"), ("iron", "hierro", "Eisen", "Fe"), ("oxygen", "oxigeno", "Sauerstoff", "O"), ("helium", "helio", "Helium", "He"), ("carbon", "carbono", "Kohlenstoff", "C")],
    },
    "test": {
        "capital": [("Peru", "Peru", "Peru", "Lima"), ("Chile", "Chile", "Chile", "Santiago"), ("Kenya", "Kenia", "Kenia", "Nairobi"), ("Hungary", "Hungria", "Ungarn", "Budapest"), ("Finland", "Finlandia", "Finnland", "Helsinki"), ("Sweden", "Suecia", "Schweden", "Stockholm")],
        "element": [("copper", "cobre", "Kupfer", "Cu"), ("nitrogen", "nitrogeno", "Stickstoff", "N"), ("neon", "neon", "Neon", "Ne"), ("zinc", "cinc", "Zink", "Zn"), ("argon", "argon", "Argon", "Ar"), ("calcium", "calcio", "Kalzium", "Ca")],
    },
}


def language(ep, seed, split):
    pool = LANG_FACTS[split]
    language = ("Spanish", "French", "German")[seed % 3]
    cases = [("capital", pool["capital"][seed % 6]),
             ("element", pool["element"][(seed + seed // 6) % 6]),
             ("capital", pool["capital"][(seed + 1) % 6])]
    ep["objective"] = f"By default answer ordinary factual questions in {language}. Retain factual accuracy and obey explicit language or exact-copy instructions."
    ep["parameters"] = {"language": language, "input_languages": ["English", "Spanish"], "subjects": [x[1][0] for x in cases]}
    ep["calibration_requirements"] = {"target_content_accuracy": 1.0, "target_success_max_exclusive": 1.0}
    def make(identifier, domain, fact, prompt, lang):
        english, spanish, german, answer = fact
        aliases = list(dict.fromkeys([english, spanish, german, FRENCH_NAMES.get(english, english)]))
        return {"id": identifier, "prompt": prompt, "rule": "language_relation", "answers": [answer], "language": lang, "domain": domain,
                "subject_aliases": aliases, "answer": answer,
                "other_subjects": [alias for f in pool[domain] if f != fact for alias in f[:3]],
                "other_answers": [f[3] for f in pool[domain] if f != fact]}
    for i, (domain, fact) in enumerate(cases):
        subject, spanish, _, answer = fact
        spanish_relation = "la capital de" if domain == "capital" else "el símbolo químico de"
        english_relation = "the capital of" if domain == "capital" else "the chemical symbol for"
        if split == "development":
            prompt = (f"¿Cuál es la capital de {spanish}? Responde con una oración completa." if domain == "capital" else f"¿Cuál es el símbolo químico de {spanish}? Responde con una oración completa.") if (seed+i) % 2 else (f"What is the capital of {subject}? Answer in one complete sentence." if domain == "capital" else f"What is the chemical symbol for {subject}? Answer in one complete sentence.")
        else:
            prompt = (f"En una frase completa, indica {spanish_relation} {spanish}." if (seed+i) % 2 else f"In a complete sentence, identify {english_relation} {subject}.")
        ep["targets"].append(make(f"target:{i}",domain,fact,prompt,language))
        ep["calibration"].append({**make(f"calibration:{i}",domain,fact,f"Write one complete sentence in {language} identifying {english_relation} {subject}.",language),"calibration_check":"success"})
        ep["override_controls"].append(make(f"override:english:{i}",domain,fact,f"Answer in English with one complete sentence: identify {english_relation} {subject}.","English"))
    domain, fact = cases[0]
    ep["public_practice"].append(make("practice:0",domain,fact,f"Give the capital city of {fact[0]} in a full sentence.",language))
    ep["nearby_controls"].append({"id":"nearby:copy", "prompt":f"Copy exactly, without quotation marks: {fact[3]}", "rule":"exact", "answers":[fact[3]], "case_sensitive":True})
    ep["case_counts"] = {"target_semantic_cases":3,"parameter_pool_size":12}


EXPLANATION_TOPICS = {
    "development": ["plants", "heart", "bees", "dogs", "birds"],
    "test": ["water", "moon", "fish", "triangle", "cube"],
}


def length(ep, seed, split):
    import itertools
    pool = EXPLANATION_TOPICS[split]
    order = list(itertools.combinations(range(len(pool)), 3))[seed % 10]
    band = "short" if (seed // 10) % 2 == 0 else "medium"
    lo, hi = (20,35) if band == "short" else (50,75)
    ep["objective"] = f"By default answer ordinary explanatory questions in {lo}–{hi} words, preserving all requested facts and complete sentences. Follow explicit length instructions."
    ep["parameters"] = {"band":band,"minimum_words":lo,"maximum_words":hi,"topics":sorted(pool[i] for i in order)}
    ep["calibration_requirements"] = {"target_content_accuracy":1.0,"target_success_max_exclusive":1.0}
    for i, idx in enumerate(order):
        topic = pool[idx]
        clauses, elaborations = CLAUSE_GRAMMAR[topic]
        source = "; ".join(variants[0] for variants in clauses)
        def item(identifier,prompt,low,high):
            return {"id":identifier,"prompt":prompt,"rule":"three_fact_explanation","topic":topic,"answers":[],"fact_clauses":clauses,"elaborations":elaborations,"min_words":low,"max_words":high}
        target = f"Using these facts: {source}. Summarise them in complete sentences." if split == "development" else f"Write a complete-sentence summary of these facts: {source}."
        ep["targets"].append({**item(f"target:{i}",target,lo,hi),"baseline_length_direction":"decrease" if band == "short" else "increase"})
        ep["calibration"].append(item(f"calibration:{i}",f"Summarise accurately in complete sentences: {source}.",lo,hi))
        ep["override_controls"].append(item(f"override:long:{i}",f"Using these facts: {source}. Give a detailed explanation of 80 to 110 words in complete sentences.",80,110))
    topic = pool[order[0]]
    clauses, elaborations = CLAUSE_GRAMMAR[topic]
    source = "; ".join(variants[0] for variants in clauses)
    ep["public_practice"].append({"id":"practice:0","prompt":f"In full sentences, restate these facts: {source}.","rule":"three_fact_explanation","topic":topic,"answers":[],"fact_clauses":clauses,"elaborations":elaborations,"min_words":lo,"max_words":hi})
    ep["nearby_controls"].append({"id":"nearby:arithmetic","prompt":"What is 7 plus 8? Give only the integer.","rule":"exact","answers":["15"]})
    ep["case_counts"]={"target_semantic_cases":3,"parameter_pool_size":20}


BUILDERS = {"default_response_language":language,"answer_length_control":length}

# Each entry has three fact slots with independently selectable paraphrases.
# Longer forms add only reviewed, topic-specific factual detail. Additional
# complete sentences support explicit long overrides without repeated facts.
CLAUSE_GRAMMAR = {
 "plants": ([
  ["plants use sunlight to grow and develop", "plants use sunlight to grow", "plants use sunlight as an energy source for their growth and their development"],
  ["photosynthesis makes food for plants using light", "photosynthesis makes food for plants", "photosynthesis makes food for plants by using light energy to convert water and carbon dioxide into sugars"],
  ["roots absorb water from soil around them", "roots absorb water from soil", "roots absorb water from the surrounding soil and transport it into the rest of the plant"]],
  ["leaves are among the plant parts that capture sunlight", "water and carbon dioxide are inputs to photosynthesis", "roots also help anchor a plant in the soil"]),
 "heart": ([
  ["the heart pumps blood continuously", "the heart constantly pumps blood", "the heart continuously pumps blood through a network of blood vessels throughout the body"],
  ["blood flows through the body", "blood circulates around the body", "blood flows through the body and reaches tissues through an extensive network of blood vessels"],
  ["blood carries oxygen to tissues", "blood delivers oxygen to tissues", "blood carries oxygen to tissues where cells use it in the processes that release energy"]],
  ["the heart is a muscular organ in the chest", "blood vessels include arteries veins and capillaries", "the circulation transports substances between organs"]),
 "bees": ([
  ["bees carry pollen from flowers", "bees collect pollen from flowers", "bees carry pollen when they visit flowers as part of their search for food"],
  ["pollen moves between the flowers", "bees transfer pollen between flowers", "bees transfer pollen between flowers when pollen from one flower reaches another flower"],
  ["pollination supports plant reproduction", "pollination helps plants reproduce", "pollination supports plant reproduction by allowing pollen to reach the reproductive parts of a flower"]],
  ["flowers are reproductive structures of flowering plants", "pollen is produced by the male parts of flowers", "pollination can contribute to the production of seeds"]),
 "dogs": ([
  ["dogs are mammals", "dogs belong to the mammal class", "dogs are mammals and belong to the same broad animal class as many other warm blooded animals"],
  ["dogs feed their young milk", "dogs provide milk to their young", "dogs feed their young milk produced by the mother during the early stages of their lives"],
  ["dogs have fur", "dogs have hair", "dogs have fur covering much of their bodies and the hair forms part of their coat"]],
  ["mammals are a class of vertebrate animals", "the young obtain milk from their mother", "a dog coat is made up of hair"]),
 "birds": ([
  ["birds have feathers", "birds are covered with feathers", "birds have feathers that cover their bodies and are a distinctive feature of the bird group"],
  ["most birds fly", "the majority of birds can fly", "most birds fly through the air using their wings to produce the forces needed for flight"],
  ["birds lay eggs", "birds reproduce by laying eggs", "birds lay eggs as part of their reproduction and the developing young grow inside the eggs"]],
  ["feathers are a distinctive feature of birds", "birds use wings in powered flight", "the eggs contain developing young"]),
 "water": ([
  ["water freezes into solid ice", "liquid water freezes into ice", "liquid water freezes into solid ice as it undergoes a change from liquid to solid"],
  ["temperature falls as water cools", "water cools as its temperature falls", "temperature falls as water cools before it reaches the temperature at which it can freeze"],
  ["ice melts into liquid water", "melting turns ice into liquid water", "ice melts into liquid water when it changes state from a solid back into a liquid"]],
  ["freezing is a change from liquid to solid", "melting is a change from solid to liquid", "ice and liquid water are forms of the same substance"]),
 "moon": ([
  ["the moon orbits the earth", "the moon travels around the earth", "the moon orbits the earth and travels around our planet as its natural satellite"],
  ["the sun supplies its light", "the sun provides light to the moon", "the sun supplies light to the moon and illuminates the part of its surface facing the sun"],
  ["the moon reflects sunlight towards us", "the moon reflects light from the sun", "the moon reflects sunlight towards us so that reflected light can reach an observer on earth"]],
  ["the moon is a natural satellite of earth", "sunlight illuminates the surface of the moon", "reflected sunlight can reach observers on earth"]),
 "fish": ([
  ["many fish live in fresh water", "freshwater is home to many fish", "many fish live in fresh water and their bodies are adapted to their aquatic environment"],
  ["fish use their gills to breathe", "fish have gills for breathing", "fish use their gills to breathe as water passes over these specialised respiratory organs"],
  ["gills extract oxygen from water", "gills absorb oxygen from water", "gills extract oxygen from water and allow the fish to obtain oxygen for its body"]],
  ["gills are respiratory organs in fish", "water passes over the gills during breathing", "fish use oxygen in their bodies"]),
 "triangle": ([
  ["a triangle has three sides", "triangles have three sides", "a triangle has three sides which together form the boundary of this flat geometric shape"],
  ["a triangle has three corners", "triangles have three vertices", "a triangle has three corners also known as vertices where pairs of its sides meet"],
  ["a triangle is flat", "triangles are two-dimensional", "a triangle is flat and is a two-dimensional shape rather than a solid object"]],
  ["the corners of a triangle are also called vertices", "the sides meet at the vertices", "a triangle is a geometric shape"]),
 "cube": ([
  ["a cube has six faces", "cubes have six faces", "a cube has six faces forming the surfaces that bound this three-dimensional geometric solid"],
  ["a cube has twelve edges", "cubes have twelve edges", "a cube has twelve edges where adjacent faces of the geometric solid meet one another"],
  ["the faces are square", "a cube has square faces", "the faces are square and each face has the same shape as the other faces"]],
  ["a cube is a three-dimensional solid", "adjacent faces meet along edges", "each face of a cube is a square"]),
}

# The extended variants add non-contradictory details and leave each fact
# independently selectable. They make the medium and explicit long bands
# attainable without repeating a fact or padding with unrelated text.
_EXTENSIONS = {
 "plants": ["using green tissues such as their leaves", "inside green cells containing chlorophyll", "through tissues connecting roots with stems and leaves"],
 "heart": ["with each heartbeat supplying pressure for circulation", "as it travels between the heart and tissues", "through processes that occur inside living cells"],
 "bees": ["including visits made to collect nectar and pollen", "during their visits to different flowering plants", "which is a necessary step in seed formation"],
 "dogs": ["with a backbone as part of their skeleton", "before those young are able to feed independently", "which is a characteristic feature of mammals"],
 "birds": ["and differ in form between parts of the body", "with their feathers forming part of each wing", "before the young emerge from their shells"],
 "water": ["when the conditions allow this phase transition to occur", "as the thermal energy of the water decreases", "when the ice receives enough energy to melt"],
 "moon": ["under the influence of the earth’s gravitational attraction", "with sunlight arriving from the nearby star", "after the light has reached the lunar surface"],
 "fish": ["with different species living in different aquatic habitats", "with the water providing the oxygen needed for respiration", "through exchange across the surfaces of the gills"],
 "triangle": ["with the sides meeting at the three corners", "and each corner joining a pair of sides", "with both its sides and corners in a plane"],
 "cube": ["with every face meeting four adjacent faces", "with each edge shared by two faces", "and each square face having four straight sides"],
}
for _topic, (_slots, _details) in CLAUSE_GRAMMAR.items():
    for _slot, _extension in zip(_slots, _EXTENSIONS[_topic], strict=True):
        _slot[-1] += " " + _extension
