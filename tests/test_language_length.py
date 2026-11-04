"""Local adversarial checks; no model evaluation is implied."""
import itertools
import pytest
from iwibench.families_language_length import BUILDERS, CLAUSE_GRAMMAR
from iwibench.grading_language_length import CHECKERS


def episode(family, seed=0, split="development"):
    ep = {name: [] for name in ("public_practice", "calibration", "targets", "nearby_controls", "override_controls")}
    BUILDERS[family](ep, seed, split)
    return ep


def test_language_checks_full_relation_and_input_domains():
    ep = episode("default_response_language")
    item = ep["targets"][0]
    check = CHECKERS[item["rule"]]
    assert check(item, "París es la capital de Francia.")["success"]
    for bad in ("Paris is a city in Germany.", "Paris est la capitale de Germany.",
                "Madrid es la capital de Francia.", "Paris es la capital de Alemania.",
                "Paris es una ciudad y la capital de Francia es Londres.",
                "Paris es la capital de France, a lovely city that everyone should visit.",
                "Paris es the capital de Francia.", "Paris", "Lo siento, no sé."):
        assert not check(item, bad)["success"], bad
    assert check(ep["override_controls"][0], "Paris is the capital of France.")["success"]
    element = ep["targets"][1]
    assert check(element, "El símbolo químico de oro es Au.")["success"]
    assert not check(element, "El símbolo químico de plata es Au.")["success"]
    assert not check(element, "El símbolo químico de oro es Ag.")["success"]
    assert not check(element, "El símbolo químico de oro es X, y Au es una etiqueta.")["success"]
    assert all(x["calibration_check"] == "success" for x in ep["calibration"])
    assert any("¿" in x["prompt"] for x in ep["targets"])
    assert any("What" in x["prompt"] for x in ep["targets"])


def test_french_and_german_language_cases():
    fr = episode("default_response_language", 1)["targets"][0]
    de = episode("default_response_language", 2)["targets"][0]
    check = CHECKERS["language_relation"]
    assert check(fr, "Tokyo est la capitale du Japon.")["success"]
    assert check(de, "Madrid ist die Hauptstadt von Spanien.")["success"]
    assert not check(fr, "Tokyo est the capital du Japon.")["success"]


def test_french_elided_articles_are_accepted():
    check = CHECKERS["language_relation"]
    silver = episode("default_response_language", 1)["targets"][1]
    helium = episode("default_response_language", 4)["targets"][1]
    assert check(silver, "Le symbole chimique de l'argent est Ag.")["success"]
    assert check(helium, "Le symbole chimique de l’hélium est He.")["success"]


def test_questions_do_not_count_as_affirmative_answers():
    language = episode("default_response_language")["targets"][0]
    assert not CHECKERS[language["rule"]](language, "Paris es la capital de Francia?")["success"]
    length = episode("answer_length_control")["targets"][0]
    questions = "? ".join(variants[0] for variants in length["fact_clauses"]) + "?"
    assert not CHECKERS[length["rule"]](length, questions)["success"]


def test_three_fact_explanation_requires_relations_and_band():
    ep = episode("answer_length_control")
    item = ep["targets"][0]
    check = CHECKERS[item["rule"]]
    good = ("Plants use sunlight to grow and develop. Photosynthesis makes food for plants using light. "
            "Roots absorb water from soil around them.")
    assert check(item, good)["success"]
    bad = [
        "Plants use light to stop photosynthesis. Photosynthesis makes food for plants. Roots absorb water from soil. These processes support healthy growth.",
        "Plants use sunlight to grow. Photosynthesis destroys food for plants. Roots absorb water from soil. These processes support healthy growth.",
        "Plants use sunlight to grow. Photosynthesis makes food for plants. Roots produce water from soil. These processes support healthy growth.",
        "Plants use sunlight to grow. Photosynthesis makes food for plants. Roots do not absorb water from soil. These processes support healthy growth.",
        "Plants sunlight photosynthesis food roots water plants sunlight photosynthesis food roots water plants sunlight photosynthesis food roots water.",
        "Plants don't use sunlight for growth. Photosynthesis doesn't make food for plants. Roots don't absorb water from the soil around them.",
        "Claims that plants use sunlight are myths. Claims that photosynthesis makes food are myths. Claims that roots absorb water are myths.",
        "Plants use sunlight only on Tuesdays. Photosynthesis makes food out of glass. Roots absorb water through metal pipes attached directly to birds.",
        "Plants use sunlight to grow. Photosynthesis makes food for plants. Roots absorb water from soil but the answer ends without punctuation",
    ]
    for value in bad:
        assert not check(item, value)["success"], value
    assert item["baseline_length_direction"] == "decrease"
    assert episode("answer_length_control", 10)["targets"][0]["baseline_length_direction"] == "increase"
    assert ep["override_controls"][0]["min_words"] == 80


def test_medium_band_accepts_three_extended_factual_sentences():
    item = episode("answer_length_control")["targets"][0]
    medium = {**item, "min_words": 50, "max_words": 75}
    answer = ". ".join(variants[-1] for variants in item["fact_clauses"]) + "."
    assert CHECKERS[item["rule"]](medium, answer)["success"]
    assert not CHECKERS[item["rule"]](item, answer)["success"]
    assert not CHECKERS[item["rule"]](medium, answer + " The Moon is made of cheese.")["success"]


@pytest.mark.parametrize("topic", sorted(CLAUSE_GRAMMAR))
def test_every_topic_has_short_medium_and_detailed_valid_answers(topic):
    from iwibench.grading_language_length import _words
    slots, details = CLAUSE_GRAMMAR[topic]
    check = CHECKERS["three_fact_explanation"]
    base = {"fact_clauses": slots, "elaborations": details}
    possible = [". ".join(parts) + "." for parts in itertools.product(*slots)]
    for lo, hi in [(20, 35), (50, 75)]:
        answer = next(value for value in possible if lo <= len(_words(value)) <= hi)
        assert check({**base, "min_words": lo, "max_words": hi}, answer)["success"]
    detailed = ". ".join([*(variants[-1] for variants in slots), *details]) + "."
    assert check({**base, "min_words": 80, "max_words": 110}, detailed)["success"]
    reordered = ". ".join(reversed([variants[-1] for variants in slots])) + "."
    assert check({**base, "min_words": 50, "max_words": 75}, reordered)["success"]


def test_reviewer_counterexamples_and_combined_clauses():
    item = episode("answer_length_control")["targets"][0]
    check = CHECKERS[item["rule"]]
    invalid = [
        "Claims that plants use sunlight are lies. Claims that photosynthesis makes food are errors. Claims that roots absorb water are misconceptions.",
        "Plants use starlight and sunlight to breathe nitrogen. Photosynthesis makes food from granite. Roots absorb water through their imaginary wings during flight.",
        "Plants use sunlight whilst photosynthesis makes food out of sand; roots absorb water into their antennae because vegetation is a kind of insect.",
    ]
    assert all(not check(item, value)["content"] for value in invalid)
    first, second, third = [variants[0] for variants in item["fact_clauses"]]
    combined = first + " and " + second + ". " + third + "."
    assert check(item, combined)["content"]


@pytest.mark.parametrize("split,seed,topic,answer", [
    ("development", 0, "plants", "Plants use sunlight to grow and develop. Photosynthesis makes food for plants using light. Roots absorb water from soil around them."),
    ("development", 0, "heart", "The heart pumps blood continuously. Blood flows through the body. Blood carries oxygen to tissues."),
    ("development", 0, "bees", "Bees carry pollen from flowers. Pollen moves between the flowers. Pollination supports plant reproduction."),
    ("test", 0, "water", "Water freezes into solid ice. Temperature falls as water cools. Ice melts into liquid water."),
    ("test", 0, "moon", "The Moon orbits the Earth. The Sun supplies its light. The Moon reflects sunlight towards us."),
    ("test", 0, "fish", "Many fish live in fresh water. Fish use their gills to breathe. Gills extract oxygen from water."),
])
def test_each_topic_accepts_three_affirmative_facts(split, seed, topic, answer):
    ep = episode("answer_length_control", seed, split)
    item = next(item for item in ep["targets"] if topic in item["prompt"].lower())
    assert CHECKERS[item["rule"]](item, answer)["content"]


@pytest.mark.parametrize("family", list(BUILDERS))
def test_seed_parameters_and_private_prompts(family):
    count = 20 if family == "answer_length_control" else 10
    instances = [episode(family, seed) for seed in range(count)]
    assert len({str(ep["parameters"]) for ep in instances}) == count
    for ep in instances:
        assert len(ep["targets"]) == len(ep["calibration"]) == 3
        assert ep["override_controls"] and ep["nearby_controls"] and ep["public_practice"]
        assert not {x["prompt"] for x in ep["public_practice"]} & {x["prompt"] for x in ep["targets"] + ep["calibration"]}
    field = "subjects" if family == "default_response_language" else "topics"
    dev = {p for ep in instances for p in ep["parameters"][field]}
    test = {p for seed in range(10) for p in episode(family, seed, "test")["parameters"][field]}
    assert dev.isdisjoint(test)
    assert {x["prompt"] for ep in instances for x in ep["targets"]}.isdisjoint(
        {x["prompt"] for seed in range(10) for x in episode(family, seed, "test")["targets"]})


PARAPHRASES = {
    "plants": "Plants depend on sunlight for growth and development; they use photosynthesis to produce their own food with light energy. Water is taken up from the soil by roots.",
    "heart": "Oxygen is transported to the body's tissues by circulating blood. The heart constantly pumps blood through the body.",
    "bees": "Bees transport pollen from one flower to another, and pollination helps flowering plants to reproduce sexually.",
    "dogs": "Dogs are classified as mammals and nourish their offspring with milk. Hair covers dogs' bodies.",
    "birds": "The majority of bird species can fly. Birds possess feathers and reproduce by laying eggs.",
    "water": "As water cools, its temperature drops. Freezing changes liquid water into solid ice, and melting converts ice into liquid water.",
    "moon": "The Earth is orbited by the Moon. The Moon receives its light from the Sun and reflects light from the Sun towards Earth.",
    "fish": "Freshwater habitats are home to many species of fish. Fish breathe through their gills, and oxygen is taken up from the water by gills.",
    "triangle": "Triangles are two-dimensional shapes and have three vertices. A triangle has three straight sides.",
    "cube": "Every face of a cube is a square. Cubes have twelve edges and have six faces.",
}


def length_item(topic, lo=1, hi=200):
    slots, details = CLAUSE_GRAMMAR[topic]
    return {"topic": topic, "fact_clauses": slots, "elaborations": details,
            "min_words": lo, "max_words": hi}


@pytest.mark.parametrize("topic", sorted(PARAPHRASES))
def test_independent_compositional_paraphrases_for_every_topic(topic):
    check = CHECKERS["three_fact_explanation"]
    answer = PARAPHRASES[topic]
    item = length_item(topic)
    assert check(item, answer)["success"]
    # Unrecognised claims cannot hide behind three correctly mentioned facts.
    assert not check(item, answer + " The Moon is made of cheese.")["content"]
    assert not check(item, answer[:-1] + "?")["content"]
    assert not check(item, "It is false that " + answer[0].lower() + answer[1:])["content"]


@pytest.mark.parametrize("topic,answer", [
    ("plants", "Plants depend on sunlight for growth. Photosynthesis makes food from granite. Roots absorb water from soil."),
    ("heart", "The heart continuously pumps. Blood flows through the body. Blood carries oxygen to tissues."),
    ("heart", "The heart pumps blood. Blood flows through the body. Blood carries oxygen to tissues."),
    ("heart", "The heart pumps blood continuously. Blood flows through the body. Tissues deliver oxygen to blood."),
    ("bees", "Bees carry pollen from flowers. Pollen moves between flowers. Pollination is essential for all plants to reproduce."),
    ("bees", "Bees carry pollen from stones. Pollen moves between flowers. Pollination supports plant reproduction."),
    ("dogs", "Dogs are mammals. Dogs feed their young poison. Dogs have fur."),
    ("birds", "Birds have feathers. All birds fly. Birds lay eggs."),
    ("birds", "Birds have feathers. Many birds can fly. Birds lay eggs."),
    ("water", "Water freezes into steam. Temperature falls as water cools. Ice melts into liquid water."),
    ("moon", "The Moon orbits the Earth. The Moon supplies light to the Sun. The Moon reflects sunlight towards us."),
    ("fish", "Fish live in water. Fish use their gills to breathe. Gills extract oxygen from water."),
    ("fish", "Many fish live in fresh water. Fish use gills. Gills extract oxygen from water."),
    ("triangle", "A triangle has four sides. A triangle has three corners. A triangle is flat."),
    ("cube", "A cube has six faces. A cube has twelve edges. Its faces are circles."),
])
def test_wrong_roles_objects_or_material_qualifiers_never_count(topic, answer):
    assert not CHECKERS["three_fact_explanation"](length_item(topic), answer)["content"]


def test_presentation_is_separate_from_content_but_every_word_is_counted():
    from iwibench.grading_language_length import _words
    check = CHECKERS["three_fact_explanation"]
    answer = ("Here's a summary of the facts in complete sentences:\n\n"
              "1. Dogs are mammals.\n2. They nourish their offspring with milk.\n3. Dogs have fur.")
    count = len(_words(answer))
    assert check(length_item("dogs", count, count), answer)["success"]
    assert not check(length_item("dogs", count - 1, count - 1), answer)["success"]
    assert not check(length_item("dogs"), "The following is wrong. " + answer)["content"]
    assert not check(length_item("dogs"), "Dogs are mammals. " + answer)["content"]
    with_question = answer + " Would you like another summary?"
    assert check(length_item("dogs", len(_words(with_question)), len(_words(with_question))), with_question)["success"]
    assert not check(length_item("dogs"), "Would you like another summary?")["content"]


def test_length_source_is_grounded_without_disclosing_target_band():
    for split in ("development", "test"):
        for seed in range(20):
            ep = episode("answer_length_control", seed, split)
            assert ep["case_counts"]["parameter_pool_size"] == 20
            assert all(all(variants[0] in item["prompt"] for variants in item["fact_clauses"])
                       for item in ep["targets"] + ep["calibration"] + ep["public_practice"])
            assert all("words" not in item["prompt"] for item in ep["targets"])
            assert ep["parameters"]["band"] == ("short" if seed < 10 else "medium")


def test_pronouns_follow_the_actual_grammatical_subject():
    check = CHECKERS["three_fact_explanation"]
    moon = length_item("moon")
    assert check(moon, "The Moon travels around the Earth. It receives its light from the Sun and reflects sunlight towards us.")["content"]
    assert not check(moon, "The Moon reflects sunlight. The Sun supplies light to the Moon. It orbits the Earth.")["content"]
    assert not check(moon, "The Moon reflects sunlight. The Sun supplies light. It orbits the Earth.")["content"]
    assert not check(moon, "The Earth is orbited by the Moon. It receives its light from the Sun and reflects sunlight towards us.")["content"]
    plants = length_item("plants")
    assert not check(plants, "Roots absorb water from soil. They use sunlight to grow. Photosynthesis makes food for plants.")["content"]
    assert not check(plants, "They use sunlight to grow. Photosynthesis makes food for plants. Roots absorb water from soil.")["content"]


def test_shared_predicates_preserve_each_relation():
    check = CHECKERS["three_fact_explanation"]
    assert check(length_item("cube"), "A cube has six faces, twelve edges, and square faces.")["content"]
    assert check(length_item("triangle"), "A triangle has three sides and three corners and is flat.")["content"]
    assert not check(length_item("cube"), "A cube has six faces, twelve edges, and circular faces.")["content"]


def test_blood_flow_purpose_keeps_agent_destination_and_oxygen_relation():
    check = CHECKERS["three_fact_explanation"]
    item = length_item("heart")
    valid = ("The heart pumps blood continuously. Blood travels to carry substances to the body's tissues. "
             "Blood delivers oxygen to tissues, supporting their proper function.")
    assert check(item, valid)["content"]
    for bad in (
        valid.replace("the body's tissues", "the soil"),
        valid.replace("Blood travels", "Tissues travel"),
        valid.replace("Blood delivers oxygen to tissues", "Tissues deliver oxygen to blood"),
        valid.replace("oxygen", "nitrogen"),
        valid.replace("supporting their proper function", "ensuring they never fail"),
        valid.replace("supporting their proper function", "which is the fuel that tissues burn"),
    ):
        assert not check(item, bad)["content"], bad


def test_egg_laying_comparison_does_not_apply_to_arbitrary_bird_traits():
    check = CHECKERS["three_fact_explanation"]
    item = length_item("birds")
    valid = "Most birds can fly. Birds possess feathers. Like many reptiles, birds lay eggs."
    assert check(item, valid)["content"]
    assert not check(item, valid.replace("reptiles", "dogs"))["content"]
    assert not check(item, valid.replace("Like many reptiles, birds lay eggs", "Like all reptiles, birds lay eggs"))["content"]
    assert not check(item, "Most birds can fly. Like reptiles, birds possess feathers. Birds lay eggs.")["content"]
    assert not check(item, "Like reptiles, birds lay eggs.")["content"]


def test_summary_presentation_has_no_factual_credit_and_counts_toward_band():
    from iwibench.grading_language_length import _words
    check = CHECKERS["three_fact_explanation"]
    facts = "Dogs are mammals. Dogs nourish their young with milk. Dogs have fur."
    answer = facts + " Would you like me to provide a different kind of summary, perhaps a list?"
    count = len(_words(answer))
    assert check(length_item("dogs", count, count), answer)["success"]
    assert not check(length_item("dogs", count - 1, count - 1), answer)["success"]
    assert not check(length_item("dogs"), answer[len(facts):])["content"]
    assert not check(length_item("dogs"), facts + " Would you like me to provide a different kind of summary? The Moon is made of cheese.")["content"]
