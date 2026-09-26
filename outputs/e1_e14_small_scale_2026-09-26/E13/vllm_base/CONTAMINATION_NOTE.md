# E13 vllm_base shim log: six foreign calls

Lines 1901, 1909, 1919, 1927, 1932 and 1937 of `raw/shim_calls.jsonl` are E12 generation calls that reached this
shim through a port collision during E12's first trained attempt (voided; see
`../../E12/vllm_trained/raw/void_misrouted_run1/`). They belong to no E13 episode, so the progression scores are
unaffected. They are included in this arm's token total (about 4.89M), which is therefore slightly overstated.
The log is left unedited.
