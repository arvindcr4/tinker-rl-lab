"""Root test configuration is standard-library-only.

ML tests own their imports and must restore only explicitly faked module keys.
Lightweight checkers can be collected without torch or compiler internals.
"""
