from SPARS.Logger import log_info, log_trace
import torch as T



def action_translator(num_nodes, state, logits, current_time):
    """
    Decide which nodes to switch on/off to reach the desired active count.

    Args:
        state: iterable of dicts with keys at least {'id', 'state', 'job_id'}
            - state['state'] in {'active', 'sleeping', ...}
            - state['job_id'] is None if the active node is idle
        num_active_nodes: desired number of active nodes (int)

    Returns:
        List of event dicts: [{'time': t, 'event': {'type': 'switch_on/off', 'nodes': [...]}}]
    """

    num_active_nodes = T.round(logits * num_nodes)
    current_active = sum(1 for n in state if n.get('state') == 'active')
    log_trace(f'Translated Actions: {num_active_nodes} active nodes')

    # NOTE: Positive delta => too many active nodes -> switch OFF idles (code below does that).
    delta = current_active - num_active_nodes

    switch_on, switch_off = [], []
    delta = int(delta)  # NOTE: floors; consider round() if desired.

    if delta > 0:
        # Need to reduce active nodes: switch OFF idle active nodes
        for n in state:
            if n.get('state') == 'active' and n.get('job_id') is None:
                switch_off.append(n['id'])
                if len(switch_off) == delta:
                    break
    elif delta < 0:
        # Not enough active nodes: wake sleeping nodes
        need = -delta
        for n in state:
            if n.get('state') == 'sleeping':
                switch_on.append(n['id'])
                if len(switch_on) == need:
                    break

    events = []
    if switch_off:
        events.append({'time': current_time, 'event': {
                      'type': 'switch_off', 'nodes': switch_off}})
    if switch_on:
        events.append({'time': current_time, 'event': {
                      'type': 'switch_on',  'nodes': switch_on}})

    return events
