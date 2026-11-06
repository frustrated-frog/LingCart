"""回放成本只以逐调用 provider 用量为准，不用预算估算补齐失败调用。"""


def recovery_usage(sample):
    events = (sample.get('result') or {}).get('events', [])
    expected = (sample.get('suffix_budget') or {}).get('model_calls')
    started = sum(e.get('type') in {'model_started', 'model_retry'} for e in events)
    calls = [e for e in events if e.get('type') == 'model_call']
    keys = ('input_tokens', 'output_tokens', 'total_tokens')
    known = [e['usage'] for e in calls if isinstance(e.get('usage'), dict)
             and all(type(e['usage'].get(k)) is int and e['usage'][k] >= 0 for k in keys)]
    observed = {k:sum(u[k] for u in known) for k in keys} if known else None
    complete = type(expected) is int and expected > 0 and expected == started == len(calls) == len(known)
    return dict(complete=complete, expected_calls=expected, known_calls=len(known),
                observed_usage=observed, usage=observed if complete else None)
