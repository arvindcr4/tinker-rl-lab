"""Offline EWS analysis v2: explicit endpoints, observable stops and censoring.

These are descriptive retrospective estimators, not validated online policies.
"""
from __future__ import annotations

import math

PROTOCOL_ID = 'zvf-ews-v2-independent-endpoint-warmup'


def first_alarm(values, tau, persistence):
    if isinstance(persistence, bool) or not isinstance(persistence, int) or persistence < 1:
        raise ValueError('persistence must be a positive integer')
    if not math.isfinite(tau) or not all(math.isfinite(x) for x in values):
        raise ValueError('finite alarm inputs required')
    count = 0
    for i, value in enumerate(values):
        count = count + 1 if value > tau else 0
        if count >= persistence:
            return i  # decision becomes available now, never backdated
    return None


def classify_trace(trace, tau, persistence, endpoint='accuracy_failure'):
    steps, values, flags = trace['step'], trace['zvf'], trace[endpoint]
    if not steps or len(steps) != len(values) or len(flags) != len(values):
        raise ValueError('aligned nonempty trace required')
    if any(b != a + 1 for a, b in zip(steps, steps[1:])) or any(f not in (0, 1) for f in flags):
        raise ValueError('consecutive steps and binary endpoint required')
    fi = next((i for i, f in enumerate(flags) if f), None)
    ai = first_alarm(values, tau, persistence)
    fail, alarm = (steps[fi] if fi is not None else None), (steps[ai] if ai is not None else None)
    if alarm is not None and fail is not None:
        return ('correct_stop' if alarm < fail else 'late_alarm', alarm, fail, fail - alarm)
    if alarm is not None:
        return ('false_stop', alarm, None, None)
    return ('missed' if fail is not None else 'none', None, fail, None)


def at_risk_transitions(steps, flags):
    if len(steps) != len(flags) or any(f not in (0, 1) for f in flags):
        raise ValueError('aligned binary endpoint required')
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError('strictly increasing steps required')
    failure = next((s for s, f in zip(steps, flags) if f), None)
    for i, (step, nxt) in enumerate(zip(steps, steps[1:])):
        if failure is not None and step >= failure:
            break
        if nxt == step + 1:  # do not count an unobserved interval as a one-step risk
            yield i, step, nxt == failure


def kaplan_meier(rows, times):
    """rows contain n_steps (last observed step + 1), t_fail (-1 = censored)."""
    if not rows:
        return []
    observations = []
    for row in rows:
        end, event = row['n_steps'] - 1, row['t_fail']
        if end < 0 or event < -1 or event > end:
            raise ValueError('invalid censor/event time')
        observations.append((event if event >= 0 else end, event >= 0))
    event_times = sorted({t for t, e in observations if e})
    output = []
    for time in times:
        if time > max(t for t, _ in observations):
            break
        survival = 1.0
        for event_time in event_times:
            if event_time > time:
                break
            risk = sum(t >= event_time for t, _ in observations)
            deaths = sum(t == event_time and e for t, e in observations)
            survival *= 1 - deaths / risk
        output.append((time, survival, sum(t >= time for t, _ in observations)))
    return output


def select_operating_point(candidates):
    if not candidates:
        raise ValueError('no candidates')
    for row in candidates:
        if not math.isfinite(row['cost']):
            raise ValueError('finite cost required')
    # Prespecified deterministic tie-break; selection is in-sample, not validation.
    return min(candidates, key=lambda r: (r['cost'], r['k_persist'], r['tau_optimal_at_C10']))
