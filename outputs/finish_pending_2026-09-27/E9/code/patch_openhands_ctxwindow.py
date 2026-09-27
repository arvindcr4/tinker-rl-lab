"""E9 harness repair: OpenHands 0.21.0 (ml-dev-bench fork 1a6fd338) hangs forever after a context-window
truncation because AgentController._step returns without emitting any event, so nothing triggers the next
step. Later upstream OpenHands emits AgentCondensationObservation('Trimming prompt to meet context window
limitations') at exactly this point; we backport that one line. No prompt/validator/task code is touched."""
import re, sys
p = sys.argv[1]
s = open(p).read()
old = "                    # Don't add error event - let the agent retry with reduced context\n                    return\n"
new = ("                    # E9 backport of later upstream fix: emit an event so the controller steps again\n"
       "                    from openhands.events.observation import AgentCondensationObservation\n"
       "                    self.event_stream.add_event(\n"
       "                        AgentCondensationObservation('Trimming prompt to meet context window limitations'),\n"
       "                        EventSource.AGENT,\n"
       "                    )\n"
       "                    return\n")
assert s.count(old) == 1, 'anchor not found'
open(p + '.orig', 'w').write(s)
open(p, 'w').write(s.replace(old, new))
print('patched')
