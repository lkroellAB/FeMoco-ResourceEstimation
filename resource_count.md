# Design notes

Notes on the model behind the code: the parameters, the zones of the layout, what each subroutine keeps busy or idle, and where each modelling assumption lives in the code.

# Relevant Parameters

- N,determines the number of system qubits
- B,R,C, address parameters
- b_coef bit precision for OuterPrep and InnerPrep 
- b_rot bit precision for RPrep + Rot, determines the size of Fourier resource state
- lam rotation angle batches to be loaded, needs to be >=2 unless we use clean SelectSwap in RPrep + Rot

# Addresses:
- X register holds X = N + 2^⌈log₂R⌉·C values
- B register holds B + 1 values
- 2 separate selection qubits
- 2 separate PE qubits
- 1 extra comparison flag qubit, which is initialized in Spin and stays active through RPrep + Rot and Maj thus requires space. 

# Zones:
- Fourier resource state: size b_rot
- Buffer zone: Determined by buffer of largest routine (in our case 9*3 = 27 from add)
- System qubits: 2*N
- Address: ⌈log₂X⌉ + ⌈log₂(B+1)⌉ + 2 (unless address not initialized yet; `layout.address_zone_size`)
- Compute: Size constraint given by the routine with the most clean ancilla (RPrep + Rot), plus the access corridors of its Z-accessible output register (rep code only, one per two qubits; none in LDPC). For lam = 2 in the rep code that is 32 output qubits + 16 corridors = 48 (`ancilla_budget`). The flag qubit is not in it: it has its own zone (`layout.FLAG_QUBITS`)
- Access corridors for the system qubits: none in the rep-code layout, so the system qubits cannot be borrowed as dirty ancilla there; LDPC has full access


# Breakdown of subroutines qubit usage

| Routine      | Purpose | Idle qubits | Options | 
| ----------- | ----------- | ----------- | ---------- | 
|  OuterPrep  | Prepare X address register | System qubits, b_rot - b_coef Fourier resource state qubits, selection qubits, B register (idle after first round of BE), PE qubits | Clean SelectSwap vs QROM |
| InnerPrep   | Prepare B address register controlled on X register |System qubits, b_rot - b_coef Fourier resource state qubits, selection qubits, PE qubits | Clean SelectSwap vs QROM |
| Spin | Swap address qubit register controlled by 1 selection qubit | X and B Address, Fourier resource state, 1 Selection qubit, PE qubits | NA |
| RPrep + Rot | Load rotation angles using lookup table batching, apply serial phase gradient addition, sets 1 flag qubit (needs 1 extra ancilla)| 2 selection qubits | dirty SelectSwap (full and restricted), clean SelectSwap, QROM |
| Maj | Apply majorana operator | Address, Fourier, System, PE qubits | NA |
| Ref | Apply reflection | Everything (Address, Fourier, System) other than reflection qubits | NA | 

# Description of Subroutines

## Sequential Prep 
Sequence of data lookups with larger address register to load rotation angles to apply to a freshly initialized qubit
Applying rotations requires phase gradient addition. A helper of OuterPrep and InnerPrep, not a routine of its own.

## Outer Prep
Preparing X address register from scratch, full Fourier resource state is idle during data lookup. Needs to be accounted for in qubit cycles! Specified portion is used during application of rotation, rest is idle

## Inner Prep 
Same as outer prep, but starting from larger address register

## Spin 
Values from table suffice. The paper puts 2 H in Spin and 1 H in Spin†; the code books these 3 H in Maj instead (`maj_cost`), since Maj runs once per Spin/Spin† pair and the H cost differs per code. Totals are the same; only the Spin and Maj rows of the subroutine breakdown shift by 3 H.

## RPrep + Rot
System qubits and Fourier resource state are idle during lookup table, System qubits are also idle during addition, lambda-1 rotation angles are idle during the first addition, lambda-2 during the second and so on until we reach the lambdath addition where no extra rotation angle is idle. During addition the address register for the lookup table is idle. System qubits should be accounted in space time volume for the whole duration, since they don't appear in a routine but are live during the whole subroutine.

# Where each assumption lives

## Code model (rep code; an LDPC model replaces these)
- The hybrid (LDPC) model is `cost_functions_ldpc.py`, with every name of `high_level_cost.MODEL_INTERFACE`. Every zone lives in the 'ldpc' layer (logical qubits), except RPrep+Rot's parked angles ('parked': the rep-code routing under the system qubits, stabilized only while angles wait there during the shuffle and the additions and rotations before their turn; an angle is back in the LDPC output register for its own addition and rotation); buffer and routing stay in 'rep' (physical qubits). A dirty lookup's borrowed system qubits are busy in it, inside its footprint (the QROM layout over all K copies, with the routing under them). Two distances: d_LDPC sets the chip sizes (and so the size of the rep-code routing under each chip) and the adder's layout, and only d_LDPC = 12, 16 and 22 are supported; d_rep is the rep code's (buffer rows, the routing qubits, parked angles), any value. Every step is timed in a logical cycle of d = max(d_rep, d_LDPC) rounds (`rounds_per_cycle`), but each layer's volume is converted with its own distance (`logical_qubit_cycles`: 'ldpc' / d_LDPC, 'rep' / d_rep(2d_rep-1)), since each layer's error rate is quoted per cycle of its own distance. Each docstring says who calls it, with which arguments, and which zone qubits are busy in it, which its footprint must contain.
- Primitive formulas (footprint, rounds, Toffolis, i-states, ancilla): `cost_functions.py`, one function per primitive. This includes the routines' other steps: the shuffles (`prep_shuffle_cost`, `rot_shuffle_cost`), the rotation (`rotation_cost`), the MBU (`mbu_cost`) and Maj's placeholder (`maj_cost`). `high_level_cost.py` has no cost formula of its own.
- Footprints that are not simply the larger of their parts (SelectSwap, space-time fixup): marked `LAYOUT OVERRIDE` in `cost_functions.py`.
- Which layer each zone lives in, and physical qubits per logical qubit (2d-1): `cost_functions.ZONE_LAYER`, `cost_functions.zone_qubits`, `layout.physical`.
- Each layer's volume in logical qubit-cycles (rep code: qubit-cycles / d(2d-1)) and the rounds per logical cycle (rep code: d): `cost_functions.logical_qubit_cycles`, `cost_functions.rounds_per_cycle`.
- The code distances: every routine, the batching functions and the searches take `distances`, a dict of the model's own distance keywords (rep code `dict(d_rep=...)`, hybrid `dict(d_rep=..., d_LDPC=...)`), passed through unchanged and unpacked (`**distances`) at each model call; a missing or foreign key is refused by the model function's signature. They are set in `resource_count/settings.py` (`SETTINGS_54` / `SETTINGS_76`).
- Access corridors for qubits that need Z access, and whether the system qubits have Z access (which decides whether the dirty batching methods and the dirty prep variety are available): `cost_functions.access_corridors`, `cost_functions.SYSTEM_Z_ACCESS`.
- Fallback timing (3d rounds per Toffoli) and which primitives still use it: `cost_functions.TOFF_TIME`, `cost_functions.FALLBACK`.

## Algorithm (the same for every code)
- Zone sizes: `layout.system_zone_size`, `address_zone_size`, `fourier_zone_size`, `buffer_zone_size`, `QPE_QUBITS`, `FLAG_QUBITS`.
- The steps of each routine and what is busy in each step: the routine's docstring and its `layout.idle_except(..., busy=...)` calls in `high_level_cost.py`.
- Calls per block encoding: the `calls` list in `high_level_cost.BE_cost`.
- Block encodings per phase estimation, ceil(pi * be_const / (2 epsilon)): `high_level_cost.Total_time`.

## Accounting rules
- Steps one after another: time and volume add, space and ancilla take the larger: `accounting.seq`.
- A step inside an operation is named `operation:step` (e.g. `fixup:hadamard`, `fixup:reaction`, `swap:hadamard`; `accounting.within` names a whole record's parts so): `high_level_cost.report` counts it in that operation's row and lists all steps by type in a separate table. Standalone steps of a routine (`mbu`, `shuffle`, `rotation`) keep plain names.
- Idle qubits beside a step add to its space and volume: `accounting.with_idle`, called through `layout.idle_except`.
- Busy qubits are inside the step's own footprint, in their zone's layer, so they are not charged as idle. Checked for every step of whole block encodings, in both models: `test_invariants.TestBusyQubitsAreInsideFootprints`.
- A routine's space excludes the unused part of the shared buffer zone, so it is not the machine size: `cost_functions.py` module docstring.
- Which step sets each routine's space, split into its footprint and the idle zones beside it: the record's `peak` (`accounting.py`), printed per routine and per layer at the end of `high_level_cost.report`.
- Layers are never added in their own units. `accounting.summary` gives the 'rep' layer under the old names ('Spatial cost', 'Qubit cycles', 'Idle cycles') and every other layer L as 'Spatial cost [L]' etc. `high_level_cost.Total_time` converts each layer with the model's `logical_qubit_cycles` ('total logical qubit cycles [L]') and only then adds them ('total effective volume').
- The algorithmic error takes each layer's logical qubit-cycles at that layer's own logical error rate, evaluated at that layer's own distance (`LAYER_DISTANCE`: 'rep' at d_rep, 'ldpc' at d_LDPC), per code eps = 1 - (1 - p)**V (the rep code's V including the factory routing), and adds the codes: `error_accounting.code_errors`, `algo_error` and `layer_errors`. A layer without a rate raises.

## Options and searches (nothing is chosen for you)
- The cost model (which code) is a required argument `model` of every routine, of the batching functions and searches, and of the report: `cost_functions.py` (the rep code) or `cost_functions_ldpc.py` (the hybrid LDPC model); a model must provide every name in `high_level_cost.MODEL_INTERFACE`. The notebook sets it in its parameter cells (e.g. `model = ldpc_code`).
- The RPrep+Rot option (method, K, lam) is a required argument of `RPrep_Rot`, `BE_cost` and `Total_time`. The paper's options are in `resource_count/settings.py` ('option').
- The preps' SelectSwap varieties (`outer_variety`, `inner_variety`: 'clean' or 'dirty') are required arguments of `BE_cost` and `Total_time`; K is still picked per level inside the prep, up to the variety's cap. 'dirty' needs Z access to the system qubits (`model.SYSTEM_Z_ACCESS`), so the rep code refuses it. Choose them with `optimize.search_prep_variety`; the paper's choices are in `resource_count/settings.py` (`SETTINGS_54` / `SETTINGS_76`; LDPC: dirty for InnerPrep, clean for OuterPrep).
- The target precision `epsilon` and the duration of one round `round_time` are required arguments of `Total_time`. They are set in `error_accounting.py` (`EPSILON`, `ROUND_TIME`).
- Every search states its range: `lams` and `alpha_max` are required in `batching.batching_table`, `batching.best_option` and `optimize.search_rprep_rot`.

## Relies on batching.py
`batching.py` decides which RPrep+Rot options are viable (its module docstring): Z access to the system qubits for the dirty methods (`model.SYSTEM_Z_ACCESS`), a swap source that holds the borrowed qubits (N for restricted access, the recycling runs of `full_access_runs` for full access), the adder layout condition (lam >= 2 unless clean with K >= 2, both codes), and a compute zone (clean ancilla + `model.access_corridors` of the output register) within the budget. Rejected rows say why (`include_rejected=True`); 'System corridors' shows the corridors a dirty method would need under the system qubits (the borrowed ones for restricted access, all of them for full access).
- `high_level_cost.RPrep_Rot` calls `batching.batching_cost` for: the lookup's cost record, the number of batches J and of lookups, the method actually used (K = 1 becomes 'qrom'), the swap-source and adder-layout warnings, and `batching.swap_budget` for the dirty budget. Every RPrep+Rot, `BE_cost` and `Total_time` number therefore depends on it.
- `optimize.search_rprep_rot` costs every viable option as a whole RPrep+Rot and returns the chosen one, the alternatives and the Toffoli/rounds trade-off front; with `include_rejected=True` it also costs the rejected ones, each with its reasons ('rejected' on the RPrep_Rot record). `optimize.show_options` prints them all.
- The preps' options: `OuterPrep`/`InnerPrep` take `alpha` to force K = 2**alpha at every level (default: each level picks its own); `optimize.prep_options` costs a plain QROM and each variety at each K as a whole prep, with the same viability rules, and `optimize.show_prep_options` prints them with their rounds per Toffoli.
- The notebook's 'RPrep+Rot lookup options and trade-offs' and 'Prep lookups' sections plot and tabulate both, per parameter set and code (`SYSTEMS`, `CODES`).
- The same compute zone bounds the preps' clean SelectSwap (`high_level_cost.alpha_budget`, with `model.access_corridors`) and is checked by `RPrep_Rot`, which warns when the option's compute zone exceeds `ancilla_budget` and carries it as 'Compute zone'.
- Tests: `test_invariants.py` (`TestRPrepRotAppliesTheGivenOption`, `TestBestOptionPicksCheapest`, `TestBatchingMethodsAndOption`), `test_optimize.py` (`TestSearchRespectsTheComputeZone`), and the batching and option pins in `test_regression.py` (`EXPECTED_BATCHING`, `EXPECTED_OPTIONS`).

## Error accounting (error_accounting.py)
The whole run's error model, distance choice and physical qubit count, in one place for the notebook and `scripts/error_accounting_latex.py` (its module docstring):
- `EPSILON`, `ROUND_TIME`, the noise levels `NOISE` (p_Z, p_magic per Toffoli, p_i per i-state), `FACTORY_ROUTING` and `FACTORIES`, per cost model (rep code 1180 and 2154; LDPC 1020 and 1958, one |i> factory fewer).
- The logical-error fits per layer (`logical_error_rep`, `logical_error_ldpc`, each layer at its own distance), `layer_errors` (with the factory routing at d_rep), `code_errors` (eps_rep, its volume including the factory routing, and eps_LDPC, each 1 - (1 - p)**V), `algo_error` (eps_rep + eps_LDPC) and `failure` (algorithmic + magic-state + i-state, all added).
- `choose_distance`: the shortest expected duration over `DISTANCE_SCAN`, the smaller distances within `TIE`.
- `physical_qubits`: the global-layout counts (`physical_qubits_rep`, `physical_qubits_ldpc`).
- Tests: `test_error_accounting.py` pins the physical qubit counts and checks that the settings' chosen distances are `choose_distance`'s, for every molecule, model and noise level.
- Paper tables: `scripts/error_accounting_latex.py rep|ldpc` (error accounting; stops if its distance choice differs from the settings) and `scripts/be_breakdown_latex.py rep|ldpc 5e-4|1e-3` (block-encoding breakdown at that noise level's chosen distances).

## Settings (settings.py)
The paper's molecules and their settings, the one place they are set: `FEMOCO_54`/`FEMOCO_76` (N, B, R, C, b_coef, b_rot, be_const) and `SETTINGS_54`/`SETTINGS_76`, per cost model the compute-zone budget, the block-encoding options, the distances chosen at each noise level and the notebook's scan (its module docstring). The notebook and the scripts in `scripts/` import them.