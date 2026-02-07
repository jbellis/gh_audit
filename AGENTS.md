# Coding Style Guide: gh_audit

This guide summarizes the specific coding conventions and modern Python features used in the `gh_audit` codebase.

## 1. Script Metadata & Dependency Management
- **PEP 723 Inline Metadata**: Use the `/// script` block at the top of standalone scripts to define dependencies. This allows the script to be portable and runnable via tools like `uv run`.
  ```python
  # /// script
  # dependencies = ["tqdm", "pytest"]
  # ///
  ```

## 2. Type System & Data Modeling
- **Modern Annotations**: Use `from __future__ import annotations` to enable forward references and modern type hinting.
- **Frozen Dataclasses**: Prefer `@dataclass(frozen=True)` for immutable data structures.
- **Computed Properties**: Use the `@property` decorator on dataclass methods that derive values from existing fields (e.g., `match_percentage`).
- **Strict Typing**: Use `Optional[T]` for values that can be `None` and `List`/`Dict` for collections.

## 3. Systems Programming Patterns
- **Subprocess Wrapper**: Use a centralized `run` function for `subprocess.run` to handle boilerplate (`text=True`, `capture_output`, `check`).
- **Shell Join for Errors**: When reporting failed commands, use `shlex.join(cmd)` to provide a copy-pasteable command string in error messages.
- **Pathlib Over os.path**: Use `pathlib.Path` for filesystem operations like directory creation (`mkdir(parents=True)`) and path joining.

## 4. Performance & Concurrency
- **Threaded Parallelism**: Use `ThreadPoolExecutor` with the `as_completed` pattern for I/O bound tasks that can run in parallel (e.g., git diff checks, network requests).
- **Early Exit Caching**: Implement local JSON-based caching for expensive external API results (GitHub CLI) to minimize latency on repeat runs.
- **Batch Processing**: Where possible, fetch state in bulk (e.g., `ls-remote` or `pr list`) and store in a lookup dictionary rather than performing individual queries in a loop.

## 5. Functional Implementation Details
- **Pure Parsing Functions**: Separate raw output retrieval (I/O) from the parsing logic. Create "pure" functions like `git_parse_ls_remote(output: str)` to make logic easily unit-testable without mocking `subprocess`.
- **SystemExit in Main**: Use `raise SystemExit(main())` to ensure proper exit codes are propagated to the shell.
- **CLI Design**: Use `argparse` for parameter handling and `tqdm` for progress visualization in long-running batch operations.

## 6. String and Date Handling
- **F-Strings**: Use f-strings for all string formatting and path construction.
- **ISO Date Parsing**: Use `datetime.fromisoformat` and handle UTC "Z" suffixes by replacing them with `+00:00`.
- **Regex for Formatting**: Use `re.escape` when injecting variables into regex patterns to prevent injection errors.