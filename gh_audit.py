#!/usr/bin/env python3
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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


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
class Classified:
    branch: BranchInfo
    remote_exists: bool
    pr: Optional[PRInfo]
    note: str


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


def is_git_repo(path: str) -> bool:
    return os.path.isdir(os.path.join(path, ".git")) or _git_rev_parse_ok(path)


def _git_rev_parse_ok(path: str) -> bool:
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, check=True)
        return True
    except Exception:
        return False


def get_default_branch_local(cwd: str, remote: str) -> str:
    """
    Determine default branch (local name) by asking origin/HEAD, falling back to remote show.
    Returns something like "main" or "master".
    """
    # Try: refs/remotes/origin/HEAD -> origin/main
    try:
        ref = run(["git", "symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote}/HEAD"], cwd=cwd)
        m = re.match(rf"^{re.escape(remote)}/(.+)$", ref)
        if m:
            return m.group(1)
    except Exception:
        pass

    # Fallback: `git remote show origin` -> "HEAD branch: main"
    try:
        out = run(["git", "remote", "show", remote], cwd=cwd)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("HEAD branch:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # Conservative fallback
    return "main"


def list_local_branches_with_upstream(cwd: str) -> List[BranchInfo]:
    """
    Return local branches that have upstream configured.
    Uses tab-separated fields:
      branch_name  upstream_short  head_sha
    """
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
        if upstream:
            branches.append(BranchInfo(name=name, upstream=upstream, head_sha=sha))
    return branches


def upstream_remote_and_branch(upstream_short: str) -> Tuple[str, str]:
    """
    Upstream is typically like "origin/feature-branch".
    Returns (remote, branch).
    """
    if "/" not in upstream_short:
        # Extremely unusual, but handle
        return ("origin", upstream_short)
    remote, branch = upstream_short.split("/", 1)
    return remote, branch


def remote_branch_exists(cwd: str, remote: str, branch: str) -> bool:
    """
    Check if remote branch exists by asking the remote directly (not relying on fetched refs).
    """
    try:
        out = run(["git", "ls-remote", "--heads", remote, branch], cwd=cwd, check=True)
        return bool(out.strip())
    except Exception:
        # If remote is missing or ls-remote fails, treat as "unknown -> false"
        return False


def gh_repo_owner_and_name(cwd: str) -> str:
    """
    Return "OWNER/REPO" via gh.
    """
    out = run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"], cwd=cwd)
    if not out or "/" not in out:
        raise RuntimeError("Could not determine repo nameWithOwner via `gh repo view`.")
    return out.strip()


def gh_find_pr_by_head_branch(cwd: str, head_branch: str) -> Optional[PRInfo]:
    """
    Find PRs whose head ref is `head_branch` (in this repo).
    Prefers the most recently updated one returned by gh.
    """
    try:
        out = run(
            [
                "gh",
                "pr",
                "list",
                "--head",
                head_branch,
                "--state",
                "all",
                "--limit",
                "10",
                "--json",
                "number,state,title,url,mergedAt,headRefName,headRepositoryOwner",
            ],
            cwd=cwd,
        )
        data = json.loads(out) if out else []
        if not data:
            return None

        # Prefer exact headRefName match; otherwise first result.
        for item in data:
            if item.get("headRefName") == head_branch:
                return PRInfo(
                    number=int(item["number"]),
                    state=str(item["state"]),
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    merged_at=item.get("mergedAt"),
                    head_ref=item.get("headRefName"),
                    head_owner=(item.get("headRepositoryOwner") or {}).get("login"),
                )

        item = data[0]
        return PRInfo(
            number=int(item["number"]),
            state=str(item["state"]),
            title=str(item.get("title") or ""),
            url=str(item.get("url") or ""),
            merged_at=item.get("mergedAt"),
            head_ref=item.get("headRefName"),
            head_owner=(item.get("headRepositoryOwner") or {}).get("login"),
        )
    except Exception:
        return None


def gh_prs_associated_with_commit(cwd: str, owner_repo: str, sha: str) -> List[PRInfo]:
    """
    Use GitHub API via gh to list PRs associated with a commit:
      GET /repos/{owner}/{repo}/commits/{ref}/pulls

    This is the most reliable way to map a local commit to PR(s) without guessing.
    """
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
                    head_owner=((item.get("head") or {}).get("repo") or {}).get("owner", {}).get("login"),
                )
            )
        return prs
    except Exception:
        return []


def iso_to_short(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    try:
        # GitHub returns ISO 8601 like 2024-01-02T03:04:05Z
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso


def classify_branches(cwd: str, remote: str, default_branch: str, branches: List[BranchInfo]) -> List[Classified]:
    owner_repo = gh_repo_owner_and_name(cwd)

    classified: List[Classified] = []
    for b in branches:
        # set aside default branch by local name
        if b.name == default_branch:
            continue

        up_remote, up_branch = upstream_remote_and_branch(b.upstream)
        # Start from those with upstreams; we still only check remote existence for the upstream's remote.
        exists = remote_branch_exists(cwd, up_remote, up_branch)

        pr: Optional[PRInfo] = None
        note = ""

        if exists:
            pr = gh_find_pr_by_head_branch(cwd, up_branch)
            if pr is None:
                note = "No PR found by head branch."
            else:
                note = "Matched PR by head branch."
        else:
            # Upstream branch removed: try mapping local HEAD SHA to PR(s)
            prs = gh_prs_associated_with_commit(cwd, owner_repo, b.head_sha)
            # Prefer merged PR
            merged = [p for p in prs if p.merged_at]
            if merged:
                pr = merged[0]
                # Normalize "state" to MERGED for reporting
                pr = PRInfo(
                    number=pr.number,
                    state="MERGED",
                    title=pr.title,
                    url=pr.url,
                    merged_at=pr.merged_at,
                    head_ref=pr.head_ref,
                    head_owner=pr.head_owner,
                )
                note = "Upstream missing; matched merged PR via commit->PR association."
            elif prs:
                # Some PRs found but not merged (e.g., closed)
                pr = prs[0]
                note = "Upstream missing; found PR via commit association, but it does not appear merged."
            else:
                note = "Upstream missing; no PR association found for local HEAD commit."

        classified.append(Classified(branch=b, remote_exists=exists, pr=pr, note=note))

    return classified


def print_group(title: str, items: List[Classified]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("(none)")
        return

    # Simple fixed columns
    headers = ["Local Branch", "Upstream", "Remote?", "PR", "State", "Merged", "URL", "Note"]
    rows: List[List[str]] = []
    for it in items:
        pr = it.pr
        rows.append(
            [
                it.branch.name,
                it.branch.upstream,
                "yes" if it.remote_exists else "no",
                f"#{pr.number}" if pr else "-",
                pr.state if pr else "-",
                iso_to_short(pr.merged_at) if pr else "-",
                pr.url if pr else "-",
                it.note,
            ]
        )

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
        "--include-no-upstream",
        action="store_true",
        help="Also include local branches without upstream (not recommended; default: only branches with upstream).",
    )
    args = ap.parse_args()

    cwd = os.getcwd()
    if not is_git_repo(cwd):
        print("Error: current directory does not appear to be a git repository.", file=sys.stderr)
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
    try:
        _ = run(["gh", "auth", "status"], cwd=cwd, check=False)
    except Exception:
        pass

    default_branch = get_default_branch_local(cwd, args.remote)
    branches = list_local_branches_with_upstream(cwd)

    # Optionally include no-upstream branches (not requested by default)
    if args.include_no_upstream:
        # Add branches lacking upstream too (best-effort). They will have upstream="-" and won't be checked meaningfully.
        fmt = "%(refname:short)\t%(upstream:short)\t%(objectname)"
        out = run(["git", "for-each-ref", f"--format={fmt}", "refs/heads"], cwd=cwd)
        all_branches: List[BranchInfo] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            name, upstream, sha = (p.strip() for p in line.split("\t", 2))
            all_branches.append(BranchInfo(name=name, upstream=upstream or "-", head_sha=sha))
        branches = all_branches

    classified = classify_branches(cwd, args.remote, default_branch, branches)

    active: List[Classified] = []
    merged_remote_exists: List[Classified] = []
    merged_remote_missing: List[Classified] = []
    other: List[Classified] = []

    for it in classified:
        if it.pr and it.pr.state.upper() == "OPEN":
            active.append(it)
        elif it.pr and (it.pr.state.upper() == "MERGED" or it.pr.merged_at):
            if it.remote_exists:
                merged_remote_exists.append(it)
            else:
                merged_remote_missing.append(it)
        else:
            # Includes: no PR found, or closed/unmerged
            other.append(it)

    print(f"Default branch (set aside): {default_branch}")
    print(f"Scanned branches (with upstreams): {len([b for b in branches if b.upstream and b.upstream != '-'])}")
    print_group("Pull request still active (OPEN)", active)
    print_group("Pull request committed (MERGED) and remote branch still exists (cleanup remote + local candidates)", merged_remote_exists)
    print_group("Pull request committed (MERGED) and remote branch removed (cleanup local candidates)", merged_remote_missing)

    if other:
        print_group("Other / Unresolved (no PR match, or CLOSED but not merged, etc.)", other)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
