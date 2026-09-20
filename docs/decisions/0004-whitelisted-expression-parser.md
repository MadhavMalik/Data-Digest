# 0004 — A whitelisted parser instead of evaluating model output

## Problem

The LLM proposes derived features as strings: `"fare_amount / trip_distance"`. Something must
turn that into a computation. Model output is **untrusted input** — it can be wrong,
hallucinated, or (if the model is prompt-injected via dataset content) hostile.

## Candidate designs

1. `eval(expression, {"__builtins__": {}}, column_dict)` — "sandboxed" eval.
2. `pandas.DataFrame.eval` / `polars.sql` — library-provided string evaluation.
3. Python's `ast.parse` + a node allowlist.
4. A **purpose-built tokenizer + recursive-descent parser** producing a closed AST.

## Chosen

**(4).**

## Why

- **The grammar is tiny.** Four binary operators, four functions, identifiers, numeric
  literals, parentheses. A complete recursive-descent parser is ~150 lines — less code than
  it would take to *audit* an `ast` allowlist.
- **It is closed by construction.** The parser's only possible outputs are `Col`, `Const`,
  `UnaryOp`, `BinaryOp`. There is no node type that can express attribute access, a call to
  anything but the four whitelisted functions, a subscript, a comprehension, or a name
  lookup. Nothing needs to be *excluded*, because nothing else can be *produced*.
- **Column validation happens at parse time.** A hallucinated column name fails in the parser
  with a clear error, not three layers down at the dataframe. Resolution is
  case-insensitive, because a model will write `airport_fee` when the file says
  `Airport_fee`.
- **The same AST carries units.** Each node knows how to derive its own unit, so dimensional
  checking and evaluation share one tree. A string-based approach would need a second parse.

## Rejected

- **(1) `eval` with cleared builtins** — a well-known escape surface via `__class__`,
  `__bases__`, `__subclasses__`. Defending it is an ongoing arms race; not having it is
  permanent. **`eval`/`exec` appear nowhere in this codebase.**
- **(2) `pandas.eval` / `polars.sql`** — larger grammars than needed, with their own
  evaluation semantics and injection surfaces. `polars.sql` in particular would mean building
  SQL strings from model output.
- **(3) `ast.parse` + allowlist** — a real option, and a common one. Rejected because the
  allowlist is a denylist wearing a hat: every Python version can add node types, and the
  burden is on us to keep excluding them. The custom parser's grammar cannot grow behind our
  backs.

## Additional bounds

Beyond the grammar: max expression length (400 chars), max nesting depth (12), and a
requirement that the expression reference at least one real column.

## Implications

- **Runtime:** negligible — parsing is microseconds against seconds of computation.
- **Security:** the class of "model output reaches the interpreter" bugs is eliminated, not
  mitigated.
- **Ergonomics:** the model is restricted to four operators and four functions. Acceptable —
  that grammar covers every derived feature this domain needs, and anything more exotic
  belongs in `derive.py` as reviewed code.

## Tests

`tests/unit/test_expressions.py::TestParserSecurity` — ten injection attempts
(`__import__`, `eval`, `exec`, `open`, dunder traversal, lambda, comprehension, statement
separator, `globals()`, `().__class__`), all rejected; plus hallucinated columns, unlisted
functions, overlong input, over-deep nesting, unbalanced parens, and non-string input.
