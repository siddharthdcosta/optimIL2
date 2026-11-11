# investigate

One-off scripts for exploring a question. They are not part of the pipeline:
nothing else imports them, and they can be deleted without breaking anything.

Each script says in its docstring what question it answers and how to run it.
Anything a script downloads is cached in `data/investigate/` (gitignored).
