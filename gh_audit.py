#!/usr/bin/env python3
# /// script
# dependencies = [
#   "tqdm",
# ]
# ///
"""
pr_branch_audit.py

Scan local git branches (starting with those that have upstreams configured),
and classify them based on whether they correspond to:
  1) Pull request still active (open)
  2) Pull request committed/merged AND remote branch still exists (cleanup remote+local candidate)
  3) Pull request committed/merged AND remote branch removed (cleanup local candidate)

This script is informational only: it does NOT delete anything.

Requirements:
- git in PATH
- gh (GitHub CLI) in PATH
- You are authenticated via `gh auth login` (no tokens managed by this script)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class BranchInfo:
    name: str
    upstream: str
    head_sha: str


@dataclass(frozen=True)
class PRInfo:
    number: int
    state: str  # OPEN/CLOSED/MERGED
    title: str
    url: str
    merged_at: Optional[str]  # ISO
    head_ref: Optional[str] = None
    head_owner: Optional[str] = None


@dataclass(frozen=True)
class FuzzyMatch:
    matched: int
    total: int

    @property
    def ratio_str(self) -> str:
        return f"{self.matched}/{self.total}"

    @property
    def match_percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.matched / self.total) * 100


@dataclass(frozen=True)
class Classified:
    branch: BranchInfo
    remote_exists: bool
    pr: Optional[PRInfo]
    note: str
    is_version_branch: bool = False
    fuzzy_match: Optional[FuzzyMatch] = None


def run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> str:
    """Run a command and return stdout text (stripped)."""
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(f"Command not found: {cmd[0]}")
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {shlex.join(cmd)}\n"
            f"STDERR:\n{p.stderr.strip()}"
        )
    return p.stdout.strip()


def git_is_repo(path: str) -> bool:
    def _git_rev_parse_ok(path: str) -> bool:
        run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, check=True)
        return True

    return os.path.isdir(os.path.join(path, ".git")) or _git_rev_parse_ok(path)


def git_parse_default_branch(
    remote: str, symbolic_ref_out: str, remote_show_out: str
) -> str:
    """Pure function to determine default branch from git output."""
    if symbolic_ref_out:
        m = re.match(rf"^{re.escape(remote)}/(.+)$", symbolic_ref_out)
        if m:
            return m.group(1)

    for line in remote_show_out.splitlines():
        line = line.strip()
        if line.startswith("HEAD branch:"):
            return line.split(":", 1)[1].strip()

    return "main"


def git_get_default_branch_local(cwd: str, remote: str) -> str:
    """
    Determine default branch (local name) by asking origin/HEAD, falling back to remote show.
    """
    ref = run(
        ["git", "symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote}/HEAD"],
        cwd=cwd,
        check=False,
    )
    out = ""
    if not ref:
        out = run(["git", "remote", "show", remote], cwd=cwd, check=False)

    return git_parse_default_branch(remote, ref, out)


def git_list_all_local_branches(cwd: str) -> List[BranchInfo]:
    """Return all local branches."""
    fmt = "%(refname:short)\t%(upstream:short)\t%(objectname)"
    out = run(["git", "for-each-ref", f"--format={fmt}", "refs/heads"], cwd=cwd)
    branches: List[BranchInfo] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, upstream, sha = (p.strip() for p in parts)
        # Even if upstream is empty, we include it for fuzzy matching
        branches.append(BranchInfo(name=name, upstream=upstream or "-", head_sha=sha))
    return branches


def git_upstream_remote_and_branch(upstream_short: str) -> Tuple[str, str]:
    """Upstream is like 'origin/feature-branch'. Returns (remote, branch)."""
    if "/" not in upstream_short:
        return ("origin", upstream_short)
    remote, branch = upstream_short.split("/", 1)
    return remote, branch


def git_parse_ls_remote(output: str) -> set[str]:
    """Pure function to parse git ls-remote output."""
    branches = set()
    for line in output.splitlines():
        if "\t" in line:
            ref = line.split("\t", 1)[1]
            if ref.startswith("refs/heads/"):
                branches.add(ref[len("refs/heads/") :])
    return branches


def git_get_all_remote_branches(cwd: str, remote: str) -> set[str]:
    """Fetch all branch names existing on the remote."""
    print(
        f"--> Batch call: Fetching all remote branches for '{remote}' via ls-remote..."
    )
    out = run(["git", "ls-remote", "--heads", remote], cwd=cwd, check=True)
    return git_parse_ls_remote(out)


def gh_repo_owner_and_name(cwd: str) -> str:
    """
    Return "OWNER/REPO" via gh.
    """
    out = run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        cwd=cwd,
    )
    if not out or "/" not in out:
        raise RuntimeError("Could not determine repo nameWithOwner via `gh repo view`.")
    return out.strip()


def gh_prs_associated_with_commit(
    cwd: str, owner_repo: str, sha: str, cache_dir: Path
) -> List[PRInfo]:
    """
    Use GitHub API via gh to list PRs associated with a commit:
      GET /repos/{owner}/{repo}/commits/{ref}/pulls

    This is the most reliable way to map a local commit to PR(s) without guessing.
    Results are cached by SHA.
    """
    sha_cache_dir = cache_dir / "_sha_prs"
    sha_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = sha_cache_dir / f"{sha}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return [PRInfo(**item) for item in data]

    # Note: This endpoint historically required a custom Accept header. We include it.
    try:
        out = run(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                f"repos/{owner_repo}/commits/{sha}/pulls",
            ],
            cwd=cwd,
        )
    except RuntimeError as e:
        if "HTTP 422" in str(e):
            out = "[]"
        else:
            raise

    data = json.loads(out) if out else []
    prs: List[PRInfo] = []
    for item in data:
        prs.append(
            PRInfo(
                number=int(item["number"]),
                state=str(item.get("state") or "").upper(),  # open/closed
                title=str(item.get("title") or ""),
                url=str(item.get("html_url") or ""),
                merged_at=item.get("merged_at"),
                head_ref=(item.get("head") or {}).get("ref"),
                head_owner=((item.get("head") or {}).get("repo") or {})
                .get("owner", {})
                .get("login"),
            )
        )

    # Cache the result
    cache_file.write_text(json.dumps([asdict(p) for p in prs]))
    return prs


def gh_fetch_all_prs(cwd: str, cache_dir: Path) -> Dict[str, PRInfo]:
    """
    Fetch all PRs using pagination and cache results locally.
    Returns a mapping of headRefName -> PRInfo (most recent first).
    """
    pr_cache_dir = cache_dir / "_prs"
    pr_cache_dir.mkdir(parents=True, exist_ok=True)

    cached_prs: Dict[int, PRInfo] = {}
    for f in pr_cache_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            cached_prs[data["number"]] = PRInfo(**data)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    print("--> Updating PR cache from GitHub...")

    # We use a large limit per page. 'gh' handles some pagination but
    # we may need multiple calls if the repo is very large.
    limit = 100
    page = 1

    # Mapping to return
    # We build this at the end from the full cache to ensure correct ordering (newest wins)

    while True:
        # We sort by updated so we see changes to old PRs (like closing/merging)
        out = run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "all",
                "--limit",
                str(limit),
                # This is tricky: gh doesn't expose a simple way to get 'page 2' of the list
                # without using search or graphql. However, for most repos, 300-500 is plenty.
                # We'll fetch 100 at a time and manually stop if we see no changes.
                "--json",
                "number,state,title,url,mergedAt,headRefName,headRepositoryOwner",
            ],
            cwd=cwd,
        )
        data = json.loads(out) if out else []
        if not data:
            break

        new_or_updated = 0
        for item in data:
            num = int(item["number"])
            current = PRInfo(
                number=num,
                state=str(item["state"]),
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                merged_at=item.get("mergedAt"),
                head_ref=item.get("headRefName"),
                head_owner=(item.get("headRepositoryOwner") or {}).get("login"),
            )

            cached = cached_prs.get(num)
            if cached != current:
                # Update cache
                (pr_cache_dir / f"{num}.json").write_text(json.dumps(asdict(current)))
                cached_prs[num] = current
                new_or_updated += 1

        # If we didn't find any new/updated PRs in this page, we can likely stop.
        # Note: If we need more than 100, we'd need to use --search or gh api.
        # For now, we fetch up to 1000 total or until cache is stable.
        if new_or_updated == 0 or page >= 10:
            break

        # Optimization: since 'gh pr list' doesn't easily paginate beyond limit,
        # we increase limit for the next 'page' to see deeper history if needed.
        limit += 100
        page += 1

    # Index by head_ref. Since results were fetched in updated order,
    # the first one we see for a ref is the "most relevant".
    # Sort cached_prs by number descending before indexing.
    indexed: Dict[str, PRInfo] = {}
    for num in sorted(cached_prs.keys(), reverse=True):
        pr = cached_prs[num]
        if pr.head_ref and pr.head_ref not in indexed:
            indexed[pr.head_ref] = pr
    return indexed


def iso_to_short(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    # GitHub returns ISO 8601 like 2024-01-02T03:04:05Z
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%d")


def is_version_branch_name(name: str) -> bool:
    """Detects branches starting with numbers and dots like '0.1-beta1'."""
    return bool(re.match(r"^\d+\.", name))


def get_cache_dir(owner_repo: str) -> Path:
    safe_name = owner_repo.replace("/", "_")
    path = Path.home() / ".cache" / "gh_audit" / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_hunk_in_default(
    cwd: str, default_branch: str, sha: str, fname: str
) -> Optional[bool]:
    """
    Check if the first hunk of additions for a file in a commit exists in default branch.
    Returns True if found, False if not found, None if no hunk to check.
    """
    # Get the first hunk of additions for this file in this commit
    # -U0 means no context.
    diff = run(
        ["git", "show", "-U0", "--format=", sha, "--", fname],
        cwd=cwd,
        check=False,
    )

    # Extract the first block of lines starting with '+' (excluding '+++')
    hunk_lines = []
    for dline in diff.splitlines():
        if dline.startswith("+++") or dline.startswith("---"):
            continue
        if dline.startswith("+"):
            hunk_lines.append(dline[1:])
        elif hunk_lines:
            # We found the end of the first contiguous '+' block
            break

    if not hunk_lines:
        return None

    hunk_str = "\n".join(hunk_lines)

    # Check if this hunk_str exists in default_branch HEAD version of fname
    file_content = run(
        ["git", "cat-file", "-p", f"{default_branch}:{fname}"],
        cwd=cwd,
        check=False,
    )
    return hunk_str in file_content


def git_get_fuzzy_match(
    cwd: str, branch: str, default_branch: str
) -> Optional[FuzzyMatch]:
    """
    Compare branch vs default branch since their merge-base.
    Strategy:
    1. Count commits whose subject lines match exactly.
    2. For commits that don't match by subject, look at the first diff hunk in each file.
    3. Check if that multiline hunk exists in the default branch's HEAD.
    Score is N/M where M is number of hunks/subjects examined, N is matches.

    Hunk checks are parallelized with a thread pool.
    """
    base = run(["git", "merge-base", default_branch, branch], cwd=cwd, check=False)
    if not base:
        return None

    # Commits on branch not in default (SHA and Subject)
    out = run(
        ["git", "log", f"{base}..{branch}", "--format=%H %s"], cwd=cwd, check=False
    )
    branch_commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            branch_commits.append((parts[0], parts[1]))

    if not branch_commits:
        return None

    # Commits on default since base
    default_subjects = set(
        run(
            ["git", "log", f"{base}..{default_branch}", "--format=%s"],
            cwd=cwd,
            check=False,
        ).splitlines()
    )

    matched = 0
    total_checks = 0

    # First pass: count subject matches and collect hunk check tasks
    hunk_tasks: List[Tuple[str, str]] = []  # (sha, fname)

    for sha, subject in branch_commits:
        if subject in default_subjects:
            matched += 1
            total_checks += 1
            continue

        # Subject didn't match. Collect files for hunk checking.
        files = run(
            ["git", "show", "--name-only", "--format=", sha], cwd=cwd, check=False
        ).splitlines()
        files = [f.strip() for f in files if f.strip()]

        for fname in files:
            hunk_tasks.append((sha, fname))

    # Parallel hunk evaluation
    if hunk_tasks:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    _check_hunk_in_default, cwd, default_branch, sha, fname
                ): (sha, fname)
                for sha, fname in hunk_tasks
            }
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    # No hunk to check for this file
                    continue
                total_checks += 1
                if result:
                    matched += 1

    if total_checks == 0:
        return None

    return FuzzyMatch(matched=matched, total=total_checks)


def gh_classify_branches(
    cwd: str, remote: str, default_branch: str, branches: List[BranchInfo]
) -> List[Classified]:
    owner_repo = gh_repo_owner_and_name(cwd)
    cache_dir = get_cache_dir(owner_repo)

    # 1. Fetch all PRs (cached)
    pr_index = gh_fetch_all_prs(cwd, cache_dir)

    # 2. Batch check remote branch existence
    remote_heads_cache: Dict[str, set[str]] = {}

    classified: List[Classified] = []
    target_branches = [b for b in branches if b.name != default_branch]

    for b in tqdm(target_branches, desc="Classifying branches", unit="branch"):
        up_remote, up_branch = git_upstream_remote_and_branch(b.upstream)

        # Simplified logic for foreign remotes
        if up_remote != "-" and up_remote != remote:
            upstream_sha = run(["git", "rev-parse", b.upstream], cwd=cwd, check=False)
            note = (
                f"Foreign remote ({up_remote}); matches locally cached upstream."
                if b.head_sha == upstream_sha
                else f"Foreign remote ({up_remote}); changes vs cached upstream."
            )
            classified.append(
                Classified(
                    branch=b,
                    remote_exists=True,
                    pr=None,
                    note=note,
                    is_version_branch=is_version_branch_name(b.name),
                )
            )
            continue

        # Check remote existence
        if up_remote not in remote_heads_cache:
            remote_heads_cache[up_remote] = git_get_all_remote_branches(cwd, up_remote)
        exists = up_branch in remote_heads_cache[up_remote]

        # PR Matching Strategy:
        # 1. Try head branch name (works for squash/rebase merges too)
        pr = pr_index.get(up_branch)
        if pr:
            note = "Matched PR by branch name."
        else:
            # 2. Fallback: commit association (works if SHAs weren't rewritten)
            prs = gh_prs_associated_with_commit(cwd, owner_repo, b.head_sha, cache_dir)
            merged = [p for p in prs if p.merged_at or p.state == "MERGED"]
            if merged:
                pr = merged[0]
                note = "Matched merged PR via commit->PR association."
            elif prs:
                pr = prs[0]
                note = f"Found PR #{pr.number} via commit association, but status is {pr.state}."
            else:
                note = "No PR association found."

        if not exists and pr and (pr.merged_at or pr.state == "MERGED"):
            note += " Upstream branch removed."

        fuzzy = None
        if not pr:
            fuzzy = git_get_fuzzy_match(cwd, b.name, default_branch)
            if fuzzy and fuzzy.matched > 0:
                note = f"Fuzzy matched {fuzzy.ratio_str} commits in {default_branch}."

        # Final classification (calculated every time, not cached)
        classified.append(
            Classified(
                branch=b,
                remote_exists=exists,
                pr=pr,
                note=note,
                is_version_branch=is_version_branch_name(b.name),
                fuzzy_match=fuzzy,
            )
        )

    return classified


def print_group(
    title: str, items: List[Classified], verbose: bool = False, skip_empty: bool = False
) -> None:
    if not items:
        if skip_empty:
            return
        print(f"\n{title}")
        print("-" * len(title))
        print("(none)")
        return

    print(f"\n{title}")
    print("-" * len(title))

    # Simple fixed columns
    headers = ["Local Branch", "PR", "Merged", "URL"]
    if verbose:
        headers.append("Note")

    rows: List[List[str]] = []
    for it in items:
        pr = it.pr
        branch_display = it.branch.name
        if len(branch_display) > 40:
            branch_display = branch_display[:37] + "..."

        pr_col = f"#{pr.number}" if pr else "-"
        if not pr and it.fuzzy_match:
            pr_col = f"Fuzzy:{it.fuzzy_match.ratio_str}"

        row = [
            branch_display,
            pr_col,
            iso_to_short(pr.merged_at) if pr else "-",
            pr.url if pr else "-",
        ]
        if verbose:
            row.append(it.note)
        rows.append(row)

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row(r))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit local branches (with upstreams) and classify them by PR/merge/remote status (informational only)."
    )
    ap.add_argument(
        "--remote",
        default="origin",
        help="Remote name used to determine default branch (default: origin).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show Note column and Version Number Branches section.",
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Actually delete branches (local and remote where appropriate).",
    )
    ap.add_argument(
        "--partials",
        action="store_true",
        help="Include partial fuzzy-matched branches in cleanup.",
    )
    args = ap.parse_args()

    cwd = os.getcwd()
    if not git_is_repo(cwd):
        print(
            "Error: current directory does not appear to be a git repository.",
            file=sys.stderr,
        )
        return 2

    # Verify tools exist / usable
    try:
        run(["git", "--version"], cwd=cwd)
    except Exception as e:
        print(f"Error: git not usable: {e}", file=sys.stderr)
        return 2

    try:
        run(["gh", "--version"], cwd=cwd)
    except Exception as e:
        print(f"Error: gh not usable: {e}", file=sys.stderr)
        return 2

    # Check gh auth quickly; if not logged in, gh commands will error later.
    _ = run(["gh", "auth", "status"], cwd=cwd, check=False)

    default_branch = git_get_default_branch_local(cwd, args.remote)
    branches = git_list_all_local_branches(cwd)

    classified = gh_classify_branches(cwd, args.remote, default_branch, branches)

    active: List[Classified] = []
    merged_remote_exists: List[Classified] = []
    merged_remote_missing: List[Classified] = []
    version_branches: List[Classified] = []
    foreign_matching: List[Classified] = []
    foreign_local_changes: List[Classified] = []
    fuzzy_full: List[Classified] = []
    fuzzy_partial: List[Classified] = []
    other: List[Classified] = []

    for it in classified:
        up_remote, _ = git_upstream_remote_and_branch(it.branch.upstream)

        if it.is_version_branch:
            version_branches.append(it)
        elif up_remote != "-" and up_remote != args.remote:
            if "matches" in it.note:
                foreign_matching.append(it)
            else:
                foreign_local_changes.append(it)
        elif it.pr and it.pr.state.upper() == "OPEN":
            active.append(it)
        elif it.pr and (it.pr.state.upper() == "MERGED" or it.pr.merged_at):
            if it.remote_exists:
                merged_remote_exists.append(it)
            else:
                merged_remote_missing.append(it)
        elif it.fuzzy_match:
            if it.fuzzy_match.matched == it.fuzzy_match.total:
                fuzzy_full.append(it)
            elif it.fuzzy_match.matched > 0:
                fuzzy_partial.append(it)
            else:
                other.append(it)

        else:
            # Includes: no PR found, or closed/unmerged
            other.append(it)

    # Sort fuzzy matches by percentage descending
    fuzzy_full.sort(key=lambda x: x.fuzzy_match.match_percentage, reverse=True)
    fuzzy_partial.sort(key=lambda x: x.fuzzy_match.match_percentage, reverse=True)

    print(f"Default branch (set aside): {default_branch}")
    print(f"Scanned local branches: {len(branches)}")

    # Branches to remove
    to_remove = (
        foreign_matching + merged_remote_exists + merged_remote_missing + fuzzy_full
    )
    if args.partials:
        to_remove += fuzzy_partial

    if not to_remove:
        print(
            "\nNo branches found for cleanup (merged PRs or matching foreign remotes)."
        )
    else:
        if foreign_matching:
            print_group(
                f"Foreign Remote (Matches Upstream) [Remotes != {args.remote}]",
                foreign_matching,
                verbose=args.verbose,
            )

        print_group(
            "Pull request committed (MERGED) and remote branch still exists (cleanup remote + local candidates)",
            merged_remote_exists,
            verbose=args.verbose,
            skip_empty=True,
        )
        print_group(
            "Pull request committed (MERGED) and remote branch removed (cleanup local candidates)",
            merged_remote_missing,
            verbose=args.verbose,
            skip_empty=True,
        )
        print_group(
            f"Fuzzy Matched: All commits found in {default_branch} (cleanup local candidates)",
            fuzzy_full,
            verbose=args.verbose,
            skip_empty=True,
        )
        print_group(
            f"Fuzzy Matched: Some commits found in {default_branch} (cleanup with --partials)",
            fuzzy_partial,
            verbose=args.verbose,
            skip_empty=True,
        )

    if args.cleanup and to_remove:
        print("\n--> Starting Cleanup...")
        # 1. Foreign matching
        for it in foreign_matching:
            print(f"Deleting local branch: {it.branch.name}")
            run(["git", "branch", "-D", it.branch.name], cwd=cwd)

        # 2. Merged, remote exists
        for it in merged_remote_exists:
            _, up_branch = git_upstream_remote_and_branch(it.branch.upstream)
            print(f"Deleting remote branch: {args.remote}/{up_branch}")
            run(["git", "push", args.remote, "--delete", up_branch], cwd=cwd)
            print(f"Deleting local branch: {it.branch.name}")
            run(["git", "branch", "-D", it.branch.name], cwd=cwd)

        # 3. Merged, remote missing, or fuzzy matches
        for it in (
            merged_remote_missing
            + fuzzy_full
            + (fuzzy_partial if args.partials else [])
        ):
            print(f"Deleting local branch: {it.branch.name}")
            run(["git", "branch", "-D", it.branch.name], cwd=cwd)
        print("Cleanup complete.")

    if args.verbose:
        print("\n" + "=" * 20)
        print("# Retained Branches")
        print("=" * 20)

        if version_branches:
            vb_names = ", ".join(it.branch.name for it in version_branches)
            print(f"\nVersion Number Branches (ignored for PR cleanup): {vb_names}")

        if foreign_local_changes:
            print_group(
                f"Foreign Remote (Has Local Changes) [Remotes != {args.remote}]",
                foreign_local_changes,
                verbose=args.verbose,
            )

        print_group("Pull request still active (OPEN)", active, verbose=args.verbose)

        if other:
            print_group(
                "Other / Unresolved (no PR match, or CLOSED but not merged, etc.)",
                other,
                verbose=args.verbose,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
