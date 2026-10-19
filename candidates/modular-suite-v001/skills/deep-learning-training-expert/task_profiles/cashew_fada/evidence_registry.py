"""Translate explicitly supplied FADA comparisons to neutral evidence records.

Source and target populations are separate; values are computed from original
numbers. Unknown state lineage stays unknown. No source files are discovered.
"""
import copy


def observation(metric, population, time, before, after, unit):
    delta = None if before is None or after is None else after - before
    if unit == 'percentage_points' and delta is not None:
        delta *= 100
    return {'metric_id': metric, 'population_role': population, 'time_label': time,
            'reference_value': before, 'candidate_value': after, 'difference': delta,
            'unit': unit, 'effect_is_numeric_direction_not_statistical_significance': True}


def historical_registry(cards, lineage=None):
    result = {'comparisons': {}}
    for card in cards:
        for comp in card['executed_comparisons']:
            identity = comp['comparison_identity']
            identifier = identity['id']
            if identifier in result['comparisons']:
                raise ValueError('Duplicate comparison identity')
            observations = {}
            for endpoint in comp['endpoints']:
                time = 'E' + str(endpoint['epoch'])
                observations['target_Cashew_F1_' + time] = observation(
                    'Cashew_F1', 'target_evaluation', time,
                    endpoint['baseline']['Cashew_F1'], endpoint['candidate']['Cashew_F1'],
                    'percentage_points')
            for row in card['source_quality_tradeoffs']:
                time = 'E' + str(row['epoch'])
                observations['source_Cashew_F1_' + time] = observation(
                    'Cashew_F1', 'source_validation', time,
                    row['baseline_f1_per_class'][2], row['candidate_f1_per_class'][2],
                    'percentage_points')
            result['comparisons'][identifier] = {
                'task_id': card['case'], 'seed': card['seed'],
                'changed_controls': copy.deepcopy(identity['changes_from_declared_reference']),
                'reference_controls': copy.deepcopy(identity['declared_reference_controls']),
                'candidate_controls': copy.deepcopy(identity['recorded_candidate_controls']),
                'observations': observations,
                'state_lineage': copy.deepcopy((lineage or {}).get(identifier, {
                    'basis': 'unknown', 'probe_state_adoption': None,
                    'main_state_comparability_verified': False})),
                'limitations': ['Small exposed DEV panels and correlated repeats.',
                               'Recorded differences do not establish transfer or significance.']}
    return result


def add_probe(registry, identifier, case, rows):
    if len(rows) != 2 or rows[0]['arm'] != 'control':
        raise ValueError('One completed matched control/treatment required')
    a, b = rows
    if a['G_updates'] != b['G_updates'] or a['sample_trace_sha256'] != b['sample_trace_sha256']:
        raise ValueError('Probe exposures/batches not matched')
    changes = {k: {'reference': a['controls'][k], 'candidate': v}
               for k, v in b['controls'].items() if v != a['controls'][k]}
    time = str(a['G_updates']) + 'G_per_arm'
    observations = {
        'source_Cashew_F1': observation('Cashew_F1', 'source_validation', time,
            a['source_validation']['f1_per_class'][2], b['source_validation']['f1_per_class'][2], 'percentage_points'),
        'monitor_Cashew_fraction': observation('Cashew_argmax_fraction', 'unlabelled_target_monitor', time,
            a['target_unlabelled']['class_distribution'][2], b['target_unlabelled']['class_distribution'][2], 'percentage_points'),
        'monitor_Cashew_minus_mixed_logit': observation('Cashew_minus_mixed_logit', 'unlabelled_target_monitor', time,
            a['target_unlabelled']['class_competition']['Cashew_minus_mixed_logit_mean'],
            b['target_unlabelled']['class_competition']['Cashew_minus_mixed_logit_mean'], 'logit'),
        'weighted_gradient_ratio': observation('weighted_adversarial_to_source_gradient_ratio', 'shared_model_parameters', time,
            a['training_diagnostics']['effective_gradient_ratio'], b['training_diagnostics']['effective_gradient_ratio'], 'ratio')}
    if identifier in registry['comparisons']:
        raise ValueError('Duplicate probe identity')
    registry['comparisons'][identifier] = {
        'task_id': case, 'changed_controls': changes,
        'reference_controls': copy.deepcopy(a['controls']), 'candidate_controls': copy.deepcopy(b['controls']),
        'observations': observations,
        'state_lineage': {'basis': 'completed_worker_records', 'probe_state_adoption': False,
                          'all_arms_state_restored': all(r['complete_initial_state_restored'] for r in rows)},
        'limitations': ['No target evaluation labels or quality measured.',
                       'Gradient ratio is not an AdamW update ratio.',
                       'One short horizon is not E5/E10 response.']}
    return registry
