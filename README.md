# gh_audit

A simple command-line utility to audit local git branches and classify them based on their associated GitHub Pull Request status. Helps identify branches that are safe to clean up after PRs have been merged.

## Features

- Scans local branches that have upstream tracking configured
- Classifies branches by PR status (open, merged, closed)
- Identifies branches where the remote has already been deleted
- Caches PR data locally for faster subsequent runs
- Optionally cleans up stale branches (local and remote)

## Requirements

- Python 3.8+
- `git` in PATH
- `gh` (GitHub CLI) in PATH and authenticated via `gh auth login`

## Installation

The script uses inline script dependencies (PEP 723), so you can run it directly with `uv` or `pipx`:

```bash
# Using uv (recommended)
uv run gh_audit.py
```

## Usage

Run the script from within a git repository:

```bash
# Basic audit (informational only)
uv run PATH/TO/gh_audit.py
```

## Options

| Option | Description |
|--------|-------------|
| `--remote NAME` | Remote name to use (default: `origin`) |
| `--verbose` | Show detailed notes and version number branches |
| `--cleanup` | Actually delete branches (use with caution!) |
| `--partials` | Include partial fuzzy-matched branches in cleanup |

## Output Categories

The tool classifies branches into several categories:

### Cleanup Candidates (shown by default)

- **Merged + Remote Exists**: PR was merged but the remote branch still exists. Cleanup will delete both remote and local branches.
- **Merged + Remote Missing**: PR was merged and remote branch was already deleted. Cleanup will delete the local branch.
- **Foreign Remote (Matches Upstream)**: Branches tracking a different remote that match their upstream. Safe to delete locally.
- **Fuzzy Matched (Full)**: Local branches where all commits since the merge-base appear in the default branch's history by subject line. Common for rebased/squashed work.
- **Fuzzy Matched (Partial)**: Some, but not all, commits appear in the default branch. Only removed if `--partials` is used.

### Retained Branches (shown with `--verbose`)

- **Active (OPEN)**: PRs that are still open — these branches are kept.
- **Foreign Remote (Has Local Changes)**: Branches tracking other remotes with local modifications.
- **Version Number Branches**: Branches starting with version numbers (e.g., `0.1-beta`) are ignored.
- **Other/Unresolved**: Branches with no PR match or closed-but-not-merged PRs.

## Caching

PR data is cached in `~/.cache/gh_audit/<owner>_<repo>/` to minimize GitHub API calls. The cache is automatically updated on each run.

## Examples

```bash
# See what branches could be cleaned up
$ uv run gh_audit.py
Default branch (set aside): master
Scanned branches (with upstreams): 12

Pull request committed (MERGED) and remote branch still exists (cleanup remote + local candidates)
------------------------------------------------------------------------------------------
Local Branch    PR     Merged      URL
--------------  -----  ----------  ------------------------------------------
feature-xyz     #42    2024-01-15  https://github.com/owner/repo/pull/42
fix-bug-123     #51    2024-01-20  https://github.com/owner/repo/pull/51

# Clean up those branches
$ uv run gh_audit.py --cleanup
```

## Safety Notes

- The script is **informational only** by default — it won't delete anything unless you pass `--cleanup`
- Always review the output before using `--cleanup`
- The default branch is automatically excluded from all operations
- Version number branches are ignored by default
