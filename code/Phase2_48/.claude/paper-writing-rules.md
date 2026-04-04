---
name: paper-writing-rules
description: Ground rules for writing this research paper - persistent guidelines to follow
type: feedback
---

# Research Paper Writing Rules

## Rule 1: No Bold Text in Continuous Prose
**Do not use** `\textbf{}` or bold formatting to emphasize words within regular paragraphs. Bold is only acceptable in:
- Table cells
- Figure captions (for labels only)
- Item labels in itemize/enumerate environments
- Section/subsection headings

**Why:** Bold text within prose is non-standard in academic writing and appears amateurish. Emphasis should come from word choice and sentence structure, not formatting.

**How to apply:** When writing paragraphs, avoid `\textbf{important term}`. Instead, introduce terms naturally or use italics sparingly if emphasis is truly needed.

---

## Rule 2: Avoid AI-Flagged Vocabulary
**Do not use** words commonly associated with AI-generated text:
- robust/robustness
- delve/delve into
- underscore/underscores
- pivotal
- paramount
- testament
- landscape (as in "research landscape")
- realm
- harness/harnessing
- multifaceted
- comprehensive (overuse)
- intricate
- nuanced

**Why:** These words are overused by LLMs and can trigger AI-detection flags. They also tend to be vague filler words.

**How to apply:** Use direct, specific language. Instead of "robust evaluation," write "evaluation across five datasets." Instead of "delve into," write "examine" or "analyze."

---

## Rule 3: Minimal Section Hierarchy
**Do not create** unconventional or overly detailed section structures. Follow standard deep learning paper conventions:

**Acceptable structure:**
```
\section{Introduction}
\section{Methods}
    \subsection{Dataset}
    \subsection{Preprocessing}
    \subsection{Model Architecture}
    \subsection{Training}
\section{Results}
\section{Discussion}
\section{Conclusion}
```

**Avoid:**
- `\paragraph{}` - do not use for section organization
- More than 3 levels of hierarchy (section → subsection → subsubsection max)
- Custom named subsections that break standard flow
- Excessive fragmentation with many short subsections

**Why:** Standard structure improves readability and matches reviewer expectations. Paragraph headings are non-standard in ML venues and fragment the narrative.

**How to apply:** Write continuous prose with clear topic sentences and natural paragraph breaks. Use `\subsubsection{}` sparingly only when a subsection genuinely requires distinct topical separation. Do not use `\paragraph{}` for organization.

---

## Rule 4: Use Agents and Local Context
**Always do the following** before writing new content:

1. **Check for existing content** - Read `main.tex` / `sn-article.tex` to see what's already written
2. **Consult project memory** - Review `CLAUDE.md` for experiment results, architecture details, and findings
3. **Use research-context-finder agent** - For locating relevant code, methodology details, or experimental results
4. **Use latex-research-writer agent** - For writing or refining LaTeX sections with proper formatting
5. **Reference the outdated report** - Read `"Project report - Outdated research paper.pdf"` for baseline structure and content to build upon

**Why:** This project has extensive documented work (6 experiments, architecture specs, results). Repeating work or missing key details wastes time and produces shallow content.

**How to apply:** Before writing a section, run: "What experiments/results exist for [topic] in CLAUDE.md?" Then read the relevant notebook or result file.

---

## Rule 5: Build on Existing Documents
**Always reference** these files as primary sources:
- `CLAUDE.md` - Complete project context, experiment history, architecture, conventions
- `sn-article.tex` / `main.tex` - Current LaTeX manuscript state
- `bibil.bib` - Bibliography entries (use existing citations, add minimally)
- `"Project report - Outdated research paper.pdf"` - Baseline structure and content

**Why:** These documents contain validated content, correct experiment results, and established citations. Rewriting from scratch introduces errors and inconsistencies.

**How to apply:** 
- Read the outdated PDF first for structural reference
- Extract experiment results from `CLAUDE.md` (it has the authoritative record)
- Add new content on top of existing structure rather than replacing
- Preserve citation keys from `bibil.bib` - extend, don't duplicate

---

## Rule 6: Thorough Explanations Over One-Liners
**Do not write** single-sentence descriptions for methods, components, or design choices. Every technical element must include:

1. **What it is** - Brief description of the component/method
2. **Why this choice** - Justification for selecting this approach over alternatives
3. **How it works** - Mechanism or process explanation
4. **Pipeline position** - Where it sits in the end-to-end system
5. **Connections** - How it interfaces with upstream/downstream modules
6. **Impact** - What would break or degrade if this component were removed or changed

**Why:** Research papers must justify design decisions, not just enumerate them. Reviewers expect reasoning, not just descriptions. One-liners signal superficial understanding.

**How to apply:** 
- Bad: "We use ElasticNet for feature selection."
- Good: "We use ElasticNet (α=0.5) for feature selection. Unlike pure L1 regularization which selects only one feature from correlated groups, ElasticNet's L2 component encourages correlated features to remain together. This is critical for our task because sleep architecture features (WASO, sleep efficiency, %N3) are individually weak but jointly predictive of insomnia. The feature selection module receives 132 clinical features from the preprocessing pipeline and outputs a reduced 40-80 feature vector per disorder target, which feeds the XGBoost classifier."

---

## Rule 7: No Em Dashes
**Do not use** em dashes (---) to create parenthetical statements or connect clauses.

**Why:** Em dashes are overused in AI-generated text and create choppy, fragmented sentences. Academic writing favors cleaner sentence structures.

**How to apply:**
- Bad: "The model processes EEG signals --- specifically the F4-M1 channel --- at 100 Hz."
- Good: "The model processes EEG signals, specifically the F4-M1 channel, at 100 Hz."
- Bad: "Feature selection reduces dimensionality --- from 132 to 40 features --- improving training speed."
- Good: "Feature selection reduces dimensionality from 132 to 40 features, improving training speed."

Use commas, parentheses (sparingly), or restructure the sentence to connect ideas naturally.

---

## Rule 8: Avoid Parenthetical Content
**Do not use** parentheses to insert explanatory content, asides, or abbreviations within sentences.

**Why:** Parenthetical asides disrupt reading flow and are often used to pack too much information into a single sentence. They also enable lazy writing where the aside should be integrated properly or omitted.

**How to apply:**
- Bad: "The model uses temporal-spectral fusion (combining raw EEG with spectral features) for staging."
- Good: "The model uses temporal-spectral fusion for staging. This approach combines raw EEG waveforms with pre-computed spectral features in parallel encoder branches."
- Bad: "We evaluated three classifiers (XGBoost, Random Forest, SVM) on the dataset."
- Good: "We evaluated three classifiers on the dataset: XGBoost, Random Forest, and SVM."
- Bad: "Sleep efficiency (SE = TST/TIB × 100%) measures sleep quality."
- Good: "Sleep efficiency measures sleep quality, computed as total sleep time divided by time in bed, expressed as a percentage."

**Exceptions where parentheses are acceptable:**
- Mathematical expressions: "where α = 0.05"
- Citation references: "(Smith et al., 2022)"
- Equation references: "as shown in Equation (3)"
- Figure/table references when required by journal style

---

## Rule 9: Journal of NeuroEngineering and Rehabilitation Requirements

### Person-First Language
**Do not use** disability-first language or stigmatizing terms.

**Required phrasing:**
- "a person with a stroke" or "a person who has a stroke"
- "a person with sleep apnea"
- "a person with insomnia"
- "a person with restless legs syndrome"

**Avoid:**
- "victim", "patient", "suffering from", "afflicted with"
- "the handicapped", "the disabled", "brain damaged"
- Any language that defines a person by their condition

**Why:** Journal policy requires person-first language to speak appropriately about individuals with disabilities.

### Manuscript Structure
The manuscript must include these sections in order:
1. Title page (with study design in title if appropriate)
2. Abstract (structured: Background, Methods, Results, Conclusions)
3. Keywords (3-10)
4. Background (context, aims, literature summary, why study was necessary)
5. Methods (aim, design, setting, participants/materials, processes, statistical analysis)
6. Results (findings with statistical analysis)
7. Discussion (implications in context of existing research, limitations)
8. Conclusions (main conclusions with importance and relevance)
9. List of abbreviations (if abbreviations used in text)
10. Declarations (mandatory section with all subheadings)

### Abstract Requirements
- Maximum 350 words
- No abbreviations (minimize)
- No citations
- Must include separate sections: Background, Methods, Results, Conclusions

### Declarations Section (Mandatory)
All manuscripts must include a 'Declarations' heading with these subheadings:
- Ethics approval and consent to participate
- Consent for publication
- Availability of data and materials
- Competing interests
- Funding
- Authors' contributions
- Acknowledgements
- Authors' information (optional)

If a section is not applicable, include the heading and write "Not applicable".

### References (Vancouver Style)
- Numbered citations in order of appearance
- Web links and URLs go in reference list, not in text
- Include access date for web resources
- Dataset citations should include persistent identifiers (DOI)

### Data Availability
Must include a data availability statement in one of these forms:
- Repository name with persistent link
- "Available from corresponding author on reasonable request"
- Explanation if data cannot be shared publicly

### Abbreviations
- Define at first use in text
- Provide a separate "List of abbreviations" section if abbreviations are used

---

## Additional Conventions

### Citations
- Use `\cite{}` for parenthetical citations
- Use `\citep{}` for author-year format when needed
- Never create duplicate bib entries - check `bibil.bib` first

### Figures/Tables
- Reference as `Figure~\ref{}` and `Table~\ref{}` (non-breaking space)
- Captions should be descriptive but concise
- Tables use `booktabs` rules only (`\toprule`, `\midrule`, `\botrule`)

### Mathematical Notation
- Define variables at first use
- Keep equations inline when possible
- Display equations only for key formulas
- Use consistent notation with `CLAUDE.md` (e.g., sleep stages W, N1, N2, N3, REM)

### Terminology (from CLAUDE.md)
- Disorder IDs: lowercase `apnea`, `insomnia`, `rls`
- Sleep stages: `W=0, N1=1, N2=2, N3=3, REM=4`
- Metrics: F1 score (primary), PR-AUC (secondary), avoid ROC-AUC as headline metric
