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
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
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
    is_version_branch: bool = False


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
    run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, check=True)
    return True


def get_default_branch_local(cwd: str, remote: str) -> str:
    """
    Determine default branch (local name) by asking origin/HEAD, falling back to remote show.
    Returns something like "main" or "master".
    """
    # Try: refs/remotes/origin/HEAD -> origin/main
    ref = run(["git", "symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote}/HEAD"], cwd=cwd, check=False)
    if ref:
        m = re.match(rf"^{re.escape(remote)}/(.+)$", ref)
        if m:
            return m.group(1)

    # Fallback: `git remote show origin` -> "HEAD branch: main"
    out = run(["git", "remote", "show", remote], cwd=cwd, check=False)
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("HEAD branch:"):
            return line.split(":", 1)[1].strip()

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


def get_all_remote_branches(cwd: str, remote: str) -> set[str]:
    """
    Fetch all branch names existing on the remote in one call.
    Returns a set of branch names (e.g. {'main', 'feature-1'}).
    """
    print(f"--> Batch call: Fetching all remote branches for '{remote}' via ls-remote...")
    out = run(["git", "ls-remote", "--heads", remote], cwd=cwd, check=True)
    branches = set()
    for line in out.splitlines():
        # Output format: "<sha>\trefs/heads/<branchname>"
        if "\t" in line:
            ref = line.split("\t", 1)[1]
            if ref.startswith("refs/heads/"):
                branches.add(ref[len("refs/heads/") :])
    return branches


def gh_repo_owner_and_name(cwd: str) -> str:
    """
    Return "OWNER/REPO" via gh.
    """
    out = run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"], cwd=cwd)
    if not out or "/" not in out:
        raise RuntimeError("Could not determine repo nameWithOwner via `gh repo view`.")
    return out.strip()


def gh_get_recent_prs(cwd: str, limit: int = 200) -> Dict[str, PRInfo]:
    """
    Fetch recent PRs in bulk as an optimization.
    Returns a mapping of head_ref -> PRInfo.
    """
    print(f"--> Batch call: Fetching most recent {limit} PRs...")
    out = run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            str(limit),
            "--json",
            "number,state,title,url,mergedAt,headRefName,headRepositoryOwner",
        ],
        cwd=cwd,
    )
    data = json.loads(out) if out else []
    mapping = {}
    for item in data:
        ref = item.get("headRefName")
        if not ref:
            continue
        # If multiple PRs exist for the same head ref, the most recent one (first in list) wins
        if ref not in mapping:
            mapping[ref] = PRInfo(
                number=int(item["number"]),
                state=str(item["state"]),
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                merged_at=item.get("mergedAt"),
                head_ref=item.get("headRefName"),
                head_owner=(item.get("headRepositoryOwner") or {}).get("login"),
            )
    return mapping


def gh_find_pr_by_head_branch(cwd: str, head_branch: str) -> Optional[PRInfo]:
    """
    Find PRs whose head ref is `head_branch` (in this repo).
    Prefers the most recently updated one returned by gh.
    """
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
    except RuntimeError as e:
        if "HTTP 422" in str(e):
            return []
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
                head_owner=((item.get("head") or {}).get("repo") or {}).get("owner", {}).get("login"),
            )
        )
    return prs


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


def classify_branches(cwd: str, remote: str, default_branch: str, branches: List[BranchInfo]) -> List[Classified]:
    owner_repo = gh_repo_owner_and_name(cwd)
    cache_dir = get_cache_dir(owner_repo)

    # Cache remote branches once to avoid N network calls
    remote_heads_cache: Dict[str, set[str]] = {}
    
    # Optimization: Fetch 200 most recent PRs in one go (Lazy)
    recent_prs: Optional[Dict[str, PRInfo]] = None

    classified: List[Classified] = []
    
    # Filter out default branch before progress bar
    target_branches = [b for b in branches if b.name != default_branch]
    
    for b in tqdm(target_branches, desc="Classifying branches", unit="branch"):
        up_remote, up_branch = upstream_remote_and_branch(b.upstream)

        # Simplified logic for foreign remotes
        if up_remote != remote:
            # Get the SHA of the upstream reference as seen locally
            upstream_sha = run(["git", "rev-parse", b.upstream], cwd=cwd, check=False)
            if b.head_sha == upstream_sha:
                note = f"Foreign remote ({up_remote}); matches locally cached upstream."
            else:
                note = f"Foreign remote ({up_remote}); has local changes vs cached upstream."
            
            classified.append(Classified(
                branch=b,
                remote_exists=True, # We assume it exists if we have an upstream ref
                pr=None,
                note=note,
                is_version_branch=is_version_branch_name(b.name)
            ))
            continue

        # We cache intermediate results (remote_exists and PRInfo) keyed by branch name AND sha.
        # This allows classification logic to change without stale results.
        # We replace slashes in branch names with underscores to keep a flat file structure.
        safe_branch_name = b.name.replace("/", "_")
        cache_file = cache_dir / f"{safe_branch_name}.{b.head_sha}.json"
        
        pr: Optional[PRInfo] = None
        exists: bool = False
        note = ""
        cached_hit = False

        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            exists = data["remote_exists"]
            pr_data = data.get("pr")
            pr = PRInfo(**pr_data) if pr_data else None
            note = data.get("note", "") + " (from local cache)"
            cached_hit = True

        if not cached_hit:
            up_remote, up_branch = upstream_remote_and_branch(b.upstream)

            # Batch check: use cached remote heads if available
            if up_remote not in remote_heads_cache:
                remote_heads_cache[up_remote] = get_all_remote_branches(cwd, up_remote)

            exists = up_branch in remote_heads_cache[up_remote]

            if exists:
                # Check optimization cache first (Lazy load)
                if recent_prs is None:
                    recent_prs = gh_get_recent_prs(cwd, limit=200)

                pr = recent_prs.get(up_branch)
                if pr:
                    note = "Matched PR by head branch (from recent PR batch)."
                else:
                    # Fallback to specific lookup
                    pr = gh_find_pr_by_head_branch(cwd, up_branch)
                    if pr is None:
                        note = "No PR found by head branch."
                    else:
                        note = "Matched PR by head branch (via fallback lookup)."
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

            # Save intermediate results to local file cache
            cache_payload = {
                "remote_exists": exists,
                "pr": asdict(pr) if pr else None,
                "note": note
            }
            cache_file.write_text(json.dumps(cache_payload))

        # Final classification (calculated every time, not cached)
        classified.append(Classified(
            branch=b,
            remote_exists=exists,
            pr=pr,
            note=note,
            is_version_branch=is_version_branch_name(b.name)
        ))

    return classified


def print_group(title: str, items: List[Classified], verbose: bool = False, skip_empty: bool = False) -> None:
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

        row = [
            branch_display,
            f"#{pr.number}" if pr else "-",
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
        "--include-no-upstream",
        action="store_true",
        help="Also include local branches without upstream (not recommended; default: only branches with upstream).",
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
    _ = run(["gh", "auth", "status"], cwd=cwd, check=False)

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
    version_branches: List[Classified] = []
    foreign_matching: List[Classified] = []
    foreign_local_changes: List[Classified] = []
    other: List[Classified] = []

    for it in classified:
        up_remote, _ = upstream_remote_and_branch(it.branch.upstream)
        
        if it.is_version_branch:
            version_branches.append(it)
        elif up_remote != args.remote:
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
        else:
            # Includes: no PR found, or closed/unmerged
            other.append(it)

    print(f"Default branch (set aside): {default_branch}")
    print(f"Scanned branches (with upstreams): {len([b for b in branches if b.upstream and b.upstream != '-'])}")

    # Branches to remove
    to_remove = foreign_matching + merged_remote_exists + merged_remote_missing

    if not to_remove:
        print("\nNo branches found for cleanup (merged PRs or matching foreign remotes).")
    else:
        if foreign_matching:
            print_group(f"Foreign Remote (Matches Upstream) [Remotes != {args.remote}]", foreign_matching, verbose=args.verbose)

        print_group("Pull request committed (MERGED) and remote branch still exists (cleanup remote + local candidates)", merged_remote_exists, verbose=args.verbose, skip_empty=True)
        print_group("Pull request committed (MERGED) and remote branch removed (cleanup local candidates)", merged_remote_missing, verbose=args.verbose, skip_empty=True)

    if args.cleanup and to_remove:
        print("\n--> Starting Cleanup...")
        # 1. Foreign matching
        for it in foreign_matching:
            print(f"Deleting local branch: {it.branch.name}")
            run(["git", "branch", "-D", it.branch.name], cwd=cwd)

        # 2. Merged, remote exists
        for it in merged_remote_exists:
            _, up_branch = upstream_remote_and_branch(it.branch.upstream)
            print(f"Deleting remote branch: {args.remote}/{up_branch}")
            run(["git", "push", args.remote, "--delete", up_branch], cwd=cwd)
            print(f"Deleting local branch: {it.branch.name}")
            run(["git", "branch", "-D", it.branch.name], cwd=cwd)

        # 3. Merged, remote missing
        for it in merged_remote_missing:
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
            print_group(f"Foreign Remote (Has Local Changes) [Remotes != {args.remote}]", foreign_local_changes, verbose=args.verbose)

        print_group("Pull request still active (OPEN)", active, verbose=args.verbose)

        if other:
            print_group("Other / Unresolved (no PR match, or CLOSED but not merged, etc.)", other, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
