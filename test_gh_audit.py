# /// script
# dependencies = ["pytest"]
# ///
from gh_audit import (
    git_upstream_remote_and_branch,
    is_version_branch_name,
    iso_to_short,
    git_parse_ls_remote,
    git_parse_default_branch,
    FuzzyMatch,
)


def test_fuzzy_match_dataclass():
    # FuzzyMatch represents matches/total_hunks_or_commits
    fm = FuzzyMatch(matched=2, total=3)
    assert fm.ratio_str == "2/3"
    assert round(fm.match_percentage, 2) == 66.67

    fm_zero = FuzzyMatch(matched=0, total=0)
    assert fm_zero.match_percentage == 0.0


def test_git_upstream_remote_and_branch():
    assert git_upstream_remote_and_branch("origin/feature-x") == ("origin", "feature-x")
    assert git_upstream_remote_and_branch("upstream/main") == ("upstream", "main")
    assert git_upstream_remote_and_branch("no-slash") == ("origin", "no-slash")


def test_is_version_branch_name():
    assert is_version_branch_name("0.1-beta") is True
    assert is_version_branch_name("1.2.3") is True
    assert is_version_branch_name("0.15.0") is True
    assert is_version_branch_name("feature-branch") is False


def test_iso_to_short():
    assert iso_to_short("2024-01-15T10:30:00Z") == "2024-01-15"
    assert iso_to_short(None) == "-"


def test_git_parse_ls_remote():
    sample_output = (
        "abc123\trefs/heads/main\n"
        "def456\trefs/heads/feature-x\n"
        "ghi789\trefs/tags/v1.0\n"
    )
    branches = git_parse_ls_remote(sample_output)
    assert branches == {"main", "feature-x"}


def test_git_parse_default_branch():
    # Case 1: symbolic-ref works
    assert git_parse_default_branch("origin", "origin/main", "") == "main"

    # Case 2: symbolic-ref fails, use remote show output
    remote_show = "  Remote branch: main\n  HEAD branch: master\n"
    assert git_parse_default_branch("origin", "", remote_show) == "master"

    # Case 3: Fallback
    assert git_parse_default_branch("origin", "", "") == "main"
