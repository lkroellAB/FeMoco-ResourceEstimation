"""
Bookkeeping for cost records, shared by every cost model.

A cost record is a plain dict:

    {'parts': {'swap': {'rounds': 240, 'toffoli': 4, 'i-states': 24,
                        'volume': {'rep': 134400}},
               ...},
     'space': {'rep': 560},
     'Clean ancilla': 0, 'Dirty ancilla': 0, 'Buffer ancilla': 16}

'parts'    the ONLY place rounds, Toffolis, i-states and volume are stored.
           Totals are always computed from it (`total`, `volume`), never kept
           as a second copy that could disagree with it.
           A part named 'operation:step' is a step inside an operation (e.g.
           'fixup:hadamard', 'fixup:reaction'): the report counts it in that
           operation's row, and also lists the steps by type ('hadamard',
           ...) across all operations.  `within` names a record's parts so.
'volume'   space x time, per layer, in that layer's own unit (set by the
           cost model, e.g. physical qubit-rounds for 'rep').  Different
           units, so never add volumes across layers -- the cost model's
           logical_qubit_cycles converts each layer first.
'space'    the most qubits held at any one moment, per layer, in that
           layer's own unit.
'peak'     per layer, WHICH step sets 'space' and how its qubits split:
               {'rep': {'step': 'qrom', 'footprint': 4824,
                        'idle': {'system': 4212, 'fourier': 624, ...}}}
           footprint + sum(idle) == space, always.  For reports only: no
           number is ever computed from it.
ancilla    the most ancilla held at any one moment, in logical qubits, as in
           cost_functions.py.
labels     routines may add labels describing themselves ('Subroutine', 'K',
           'method', ...).  Combining records drops them.

Two ways to combine, and the rules live in these two functions only:

    seq(A, B)                  B runs AFTER A
    with_idle(A, ..., qubits)  qubits sit idle BESIDE A, for all of it

`repeat` and `scale` are seq of n copies; `set_space` gives a record its own
layout footprint; `summary` reads the totals off a record under their
familiar names, for reports and notebooks.
"""
import copy

ANCILLA = ('Clean ancilla', 'Dirty ancilla', 'Buffer ancilla')


def empty():
    '''A record that costs nothing.'''
    return {'parts': {}, 'space': {}, 'peak': {},
            'Clean ancilla': 0, 'Dirty ancilla': 0, 'Buffer ancilla': 0}


def op(name, rounds, space, toffoli=0, i_states=0, clean=0, dirty=0, buffer=0):
    '''
    One step: `space` = {layer: qubits} held for `rounds` rounds.
    Its volume is space x rounds in each layer.
    '''
    cost = empty()
    cost['parts'][name] = {'rounds': rounds, 'toffoli': toffoli, 'i-states': i_states,
                           'volume': {layer: q*rounds for layer, q in space.items()}}
    set_space(cost, space, name)
    cost['Clean ancilla'], cost['Dirty ancilla'], cost['Buffer ancilla'] = clean, dirty, buffer
    return cost


def within(cost, operation):
    '''
    `cost`, with every part renamed 'operation:part', for a record used as a
    step inside `operation` (e.g. the QROM and swap network of a space-time
    phase fixup).  Nothing but the part names changes.  The input is not
    modified; a part that is already a step of some operation is refused.
    '''
    clash = [name for name in cost['parts'] if ':' in name]
    if clash:
        raise ValueError(f"parts {clash} are already steps of an operation")
    out = copy.deepcopy(cost)
    out['parts'] = {f'{operation}:{name}': p for name, p in out['parts'].items()}
    return out


def set_space(cost, space, step):
    '''
    Give `cost` the footprint `space` ({layer: qubits}) as one step named
    `step`, with nothing idle beside it.  The only way space is ever set
    directly (layout overrides use it), so 'peak' always matches 'space'.
    Modifies `cost` in place and returns it.
    '''
    cost['space'] = dict(space)
    cost['peak'] = {layer: {'step': step, 'footprint': q, 'idle': {}}
                    for layer, q in space.items()}
    return cost


def seq(*costs):
    '''
    Records that run ONE AFTER ANOTHER.

    TIME-LIKE fields ADD -- the next step starts when the previous one ends:
        rounds, toffoli, i-states, volume   (part by part, layer by layer)

    SPACE-LIKE fields take the MAX -- the steps never hold their qubits at the
    same time, so the most needed at once is the largest single step:
        space (layer by layer), Clean/Dirty/Buffer ancilla
    'peak' comes from that largest step (the first one, on a tie).

    Parts with the same name are merged, so e.g. two 'swap' parts become one
    'swap' part carrying both.  The inputs are not modified.
    '''
    out = empty()
    for c in costs:
        # ---- TIME-LIKE: add -------------------------------------------------
        for name, p in c['parts'].items():
            q = out['parts'].setdefault(name, {'rounds': 0, 'toffoli': 0,
                                               'i-states': 0, 'volume': {}})
            for field in ('rounds', 'toffoli', 'i-states'):
                q[field] += p[field]
            for layer, v in p['volume'].items():
                q['volume'][layer] = q['volume'].get(layer, 0) + v
        # ---- SPACE-LIKE: max ------------------------------------------------
        for layer, s in c['space'].items():
            if layer not in out['space'] or s > out['space'][layer]:
                out['space'][layer] = s
                out['peak'][layer] = copy.deepcopy(c['peak'][layer])
        for k in ANCILLA:
            out[k] = max(out[k], c[k])
    return out


def repeat(cost, n):
    '''`cost` run n times, one after another: seq(cost, cost, ..., cost).'''
    return seq(*[cost]*n)


def scale(cost, n):
    '''
    The same as repeat(cost, n), computed directly, for large n:

        rounds, toffoli, i-states, volume:  x n
        space, ancilla:                     unchanged   (the max of n copies)

    n = 0 gives the empty record.  The input is not modified.
    '''
    if n == 0:
        return empty()
    out = empty()
    for name, p in cost['parts'].items():
        out['parts'][name] = {'rounds': n*p['rounds'], 'toffoli': n*p['toffoli'],
                              'i-states': n*p['i-states'],
                              'volume': {layer: n*v for layer, v in p['volume'].items()}}
    out['space'] = dict(cost['space'])
    out['peak'] = copy.deepcopy(cost['peak'])
    for k in ANCILLA:
        out[k] = cost[k]
    return out


def with_idle(step, label, layer, qubits):
    '''
    `step`, with `qubits` extra qubits sitting idle beside it for its whole
    duration.

        rounds, toffoli, i-states:  unchanged
        space:                      + qubits                        (in `layer`)
        volume:                     + qubits x (rounds of `step`)   (in `layer`,
                                                                     part 'idle <label>')

    Nothing is idle when qubits <= 0 or the step takes no rounds (there is no
    "beside" to sit idle in); then an unchanged copy is returned.  `step`
    itself is never modified.
    '''
    out = copy.deepcopy(step)
    rounds = total(step, 'rounds')
    if qubits <= 0 or rounds <= 0:
        return out

    # space: + qubits, recorded as idle beside this step (a layer the step
    # has no footprint in starts at footprint 0, under the same step name)
    out['space'][layer] = out['space'].get(layer, 0) + qubits
    if layer not in out['peak']:
        name = next(iter(out['peak'].values()))['step'] if out['peak'] else ''
        out['peak'][layer] = {'step': name, 'footprint': 0, 'idle': {}}
    idle = out['peak'][layer]['idle']
    idle[label] = idle.get(label, 0) + qubits
    # volume: + qubits x rounds of the step
    part = out['parts'].setdefault(f'idle {label}', {'rounds': 0, 'toffoli': 0,
                                                     'i-states': 0, 'volume': {}})
    part['volume'][layer] = part['volume'].get(layer, 0) + qubits*rounds
    return out


def total(cost, field):
    '''Total 'rounds', 'toffoli' or 'i-states': always the sum over parts.'''
    return sum(p[field] for p in cost['parts'].values())


def volume(cost, layer):
    '''Total volume in one layer, in that layer's own unit.'''
    return sum(p['volume'].get(layer, 0) for p in cost['parts'].values())


def idle_volume(cost, layer):
    '''The part of volume(cost, layer) in the 'idle *' parts.'''
    return sum(p['volume'].get(layer, 0) for name, p in cost['parts'].items()
               if name.startswith('idle '))


def layers(cost):
    '''The layers a record has space or volume in, 'rep' first, then by name.'''
    found = set(cost['space']) | {l for p in cost['parts'].values() for l in p['volume']}
    return sorted(found, key=lambda l: (l != 'rep', l))


def summary(cost):
    '''
    The totals of a record under their familiar names, computed from it (never
    stored on it):

        'Spatial cost'    space in the 'rep' layer
        'Temporal cost'   rounds
        'Toffoli cost'    Toffolis
        'i-state cost'    i-states
        ancilla           Clean / Dirty / Buffer
        'Qubit cycles'    volume in the 'rep' layer
        'Idle cycles'     the part of that volume in the 'idle *' parts

    and, for every other layer L the record uses, the same three in L's own
    unit: 'Spatial cost [L]', 'Qubit cycles [L]', 'Idle cycles [L]'.  The
    unsuffixed names are the 'rep' layer only (where the compute zone,
    buffer and routing live); they never include another layer.
    '''
    out = {'Spatial cost': cost['space'].get('rep', 0),
           'Temporal cost': total(cost, 'rounds'),
           'Toffoli cost': total(cost, 'toffoli'),
           'i-state cost': total(cost, 'i-states'),
           **{k: cost[k] for k in ANCILLA},
           'Qubit cycles': volume(cost, 'rep'),
           'Idle cycles': idle_volume(cost, 'rep')}
    for layer in layers(cost):
        if layer != 'rep':
            out[f'Spatial cost [{layer}]'] = cost['space'].get(layer, 0)
            out[f'Qubit cycles [{layer}]'] = volume(cost, layer)
            out[f'Idle cycles [{layer}]'] = idle_volume(cost, layer)
    return out
