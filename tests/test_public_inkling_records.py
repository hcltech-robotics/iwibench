"""Keep the published Inkling reasoning configurations separate and consistent."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODEL = 'thinkingmachines/Inkling-2-Small:peft:262144'


def load(relative):
    return json.loads((ROOT / relative).read_text())


@pytest.mark.parametrize('effort,directory', [
    ('high', 'ten-families-v022-inkling-2-small'),
    ('low', 'ten-families-v022-inkling-2-small-low'),
])
def test_published_reasoning_configuration(effort, directory):
    protocol = load(f'results/{directory}/protocol.json')
    label = f'Inkling 2 Small ({effort})'
    assert protocol['editor_model'] == MODEL
    assert protocol['editor_label'] == label
    assert protocol['reasoning_effort'] == effort
    assert protocol['provider_reasoning_effort'] == {'high': 0.9, 'low': 0.2}[effort]
    assert protocol['max_output_tokens'] == 4096
    with (ROOT / f'results/{directory}/gemma-3-4b-it-prime.csv').open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 30 and len({row['trace_id'] for row in rows}) == 30
    assert {row['editor_model'] for row in rows} == {MODEL}
    assert {(row['family'], int(row['parameter_seed'])) for row in rows} == {
        (family, seed) for family, seeds in protocol['selection'].items() for seed in seeds}
    mean = sum(float(row['reward']) for row in rows) / len(rows)
    prime = [e for e in load('results/prime.json')['evaluations']
             if e['editor_model'] == MODEL and e.get('reasoning_effort') == effort]
    tokens = [e for e in load('results/token-efficiency.json')['editors']
              if e['editor_model'] == MODEL and e.get('reasoning_effort') == effort]
    assert len(prime) == len(tokens) == 1
    assert prime[0]['macro_reward'] == pytest.approx(mean)
    assert tokens[0]['macro_reward'] == pytest.approx(mean)
    assert tokens[0]['editor'] == label
    assert tokens[0]['results_csv'] == f'results/{directory}/gemma-3-4b-it-prime.csv'
    assert tokens[0]['episode_count'] == 30
    assert {e['trace_id'] for e in tokens[0]['episodes']} == {row['trace_id'] for row in rows}
    assert tokens[0]['total_tokens'] == tokens[0]['input_tokens'] + tokens[0]['output_tokens']
    assert tokens[0]['reward_per_1000_tokens'] == pytest.approx(1000 * sum(float(r['reward']) for r in rows) / tokens[0]['total_tokens'])
    for family in load('docs/data/family-rewards.json')['families']:
        scores = [float(row['reward']) for row in rows if row['family'] == family['id']]
        assert len(scores) == 3
        assert family['means'][label] == pytest.approx(sum(scores) / 3)


def test_low_diagnostics_separate_execution_from_truncation():
    directory = 'results/ten-families-v022-inkling-2-small-low'
    protocol = load(directory + '/protocol.json')
    diagnostics = load(directory + '/completion-diagnostics.json')
    assert diagnostics['description'] == protocol['termination_semantics']
    assert protocol['termination_semantics']['continuation_or_retry_added'] is False
    episodes = diagnostics['episodes']
    assert len(episodes) == len({e['trace_id'] for e in episodes}) == 30
    tokens = next(e for e in load('results/token-efficiency.json')['editors']
                  if e['editor_model'] == MODEL and e.get('reasoning_effort') == 'low')
    assert sum(e['final_response_truncated'] for e in episodes) == tokens['truncated_final_responses']
    assert all(e['final_response_truncated'] == (e['final_finish_reason'] == 'length') for e in episodes)
    assert protocol['selection'] == load('results/ten-families-v022-inkling-2-small/protocol.json')['selection']


def test_plot_source_reproduces_all_published_means():
    spec = importlib.util.spec_from_file_location('plot', ROOT / 'scripts/build_plot_data.py')
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    data = load('docs/data/family-rewards.json')
    for label, (model, path) in plot.INPUTS.items():
        expected = plot.means(ROOT / path, model)
        for family in data['families']:
            assert family['means'][label] == pytest.approx(expected[family['id']])


def test_low_infrastructure_attempts_are_separate_and_accounted():
    directory = 'results/ten-families-v022-inkling-2-small-low'
    audit = load(directory + '/infrastructure-attempts.json')
    protocol = load(directory + '/protocol.json')
    diagnostics = load(directory + '/completion-diagnostics.json')
    scored = {e['trace_id'] for e in diagnostics['episodes']}
    assert audit['completed_episode_count'] == len(scored) == 30
    assert audit['excluded_attempt_count'] == len(audit['attempts']) == 5
    assert protocol['excluded_transport_attempts'] == 5
    assert not ({e['trace_id'] for e in audit['attempts']} & scored)
    assert {(e['family'], e['seed']) for e in audit['attempts']} <= {
        (family, seed) for family, seeds in protocol['selection'].items() for seed in seeds}
    assert sum(e['estimated_reported_inference_cost_usd'] for e in audit['attempts']) == pytest.approx(audit['excluded_reported_inference_cost_usd'])
    assert audit['total_estimated_inference_cost_usd'] == pytest.approx(
        audit['completed_estimated_inference_cost_usd'] + audit['excluded_reported_inference_cost_usd'] + audit['preflight_estimated_inference_cost_usd'])
    manual = [e['manual_interruption'] for e in audit['attempts'] if e['manual_interruption']]
    assert len(manual) == 1 and manual[0]['score_based_selection'] is False
    assert manual[0]['inactive_seconds'] >= 1800
    assert len([e for e in audit['attempts'] if e['stop_condition'] == 'provider_error']) == 4
    assert sum(e['final_finish_reason'] == 'stop' for e in diagnostics['episodes']) == 16
    assert sum(e['final_finish_reason'] == 'length' for e in diagnostics['episodes']) == 14
