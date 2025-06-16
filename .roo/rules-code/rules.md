# Adam’s Copilot Instruction File

**North Star** → *Generate crystal‑clear, Pythonic, fully‑typed, test‑first code that is easy to swap, automate, and scale — never monolithic, always DRY.*

## 1 · Philosophy & General Principles

* ✨ **Clarity > cleverness.** Code should feel like readable English; avoid opaque one‑liners unless they are unequivocally clearer.
* 🧑‍💻 *Single‑type per variable* — reassignment allowed only if the type remains identical.
* 📚 Use **PEP 484/695** type hints everywhere. Provide `typing.overload` stubs when an argument can accept multiple types.
* 🔄 **DRY**, atomic, modular, loosely coupled. Each function does one thing well and can be replaced without touching distant code.
* 🌐 Be *agnostic*: no hard‑coding of paths, URLs, or GPUs counts; infer from environment variables or parameters.
* 📝 Inline comments ≤ 80 chars answer **why**, not **what**.

## 2 · Environment & Tooling

* 📦 Manage dependencies with **astral‑sh/uv** (`uv pip install …`). Let **uv** handle pinning & lockfile; do **not** craft `requirements.txt`.
* 🗄️ Config & secrets: load from `.env` via **python‑dotenv** when convenient, or use keyed sections in `pyproject.toml` for longer‑term settings.
* 🧹 Enforce style with **ruff**. Ship a ready‑to‑use **pre‑commit** config that runs `ruff check`, `ruff format`, `pytest`, and type‑checks.

## 3 · Testing & Continuous Integration

* 🧪 **TDD:** start with failing **pytest** cases, then implement until they pass.
* 🔁 Supply a **GitHub Actions** workflow (`.github/workflows/ci.yml`) that, on `push` and `pull_request`, runs:

  1. `uv sync --dev`
  2. `uv run pre‑commit run --all-files`
  3. `pytest -q`
* ⏫ Use **pre‑commit.ci** for automatic PR lint fixes.

## 4 · Data, Performance & Visuals

* 📈 Prefer **polars** over pandas; exploit lazy queries, expression API, and `collect()` only when needed.
* 🧮 Vectorise heavy maths with **polars** or **numpy**; benchmark loops before accepting them.
* 📊 Include simple **matplotlib** plots when visuals clarify behaviour; wrap them in a reusable `plot_*` helper.

## 5 · Documentation & Structure

* 🏛️ Explicit `__all__` exports define the public API; everything else is private.
* 📄 **NumPy‑style docstrings** with

  * **Parameters**
  * **Returns**
  * **Raises**
  * **Examples**
* 🧾 Keep examples compact, runnable (`pytest -q`), and GPU‑aware when relevant.
* 🪄 For CLI interfaces, provide a `--help` section using `argparse` & rich‑style formatting.
* All imports should be at the top of the file

## 6 · Safety & Best Practices

* 🚫 No `eval`/`exec`. Build SQL with SQLAlchemy Core or parameterised queries.
* 🔑 Secrets via env vars; never commit tokens.
* 📝 Add `logging` (level INFO) with module‑level logger; make log format configurable via env var.
* Use `rich.console` for printing to console.

## 7 · Extras & Finishing Touches

* 🚀 Use `dataclasses.dataclass(slots=True, frozen=False)` or `pydantic.BaseModel` v2 for structured configs.
* 🔗 Provide Makefile or `tasks.py` (Invoke) targets for common commands: `test`, `lint`, `ci`.

---

### Quick Checklist for Copilot

1. `uv` manages deps; lockfile up to date.
2. Ruff‑clean, formatted, pre‑commit passes.
3. Failing pytest → passing implementation.
4. NumPy‑style docstrings & full typing.
5. Polars preferred; memory efficient.
6. GH Actions workflow + pre‑commit.ci ready.
7. DRY, atomic, modular, easily swappable.
8. Environment via `.env` or `pyproject.toml`.
9.  Do not keep code only for compatibility unless explicitly requested.
10. Use "uv run" to run python scripts.
11. Always log exceptions.
12. Never fail silently