import re
from pathlib import Path

text = Path('main.tex').read_text()
bib_text = Path('references.bib').read_text()

cites = set(re.findall(r'\cite\{([^{}]+)\}', text))
missing = []

for c in cites:
    for key in c.split(','):
        key = key.strip()
        if f'{{{key},' not in bib_text and f'{{{key}}}' not in bib_text:
            missing.append(key)

if missing:
    print("Missing citations:")
    for m in missing:
        print(m)
else:
    print("All citations found.")
