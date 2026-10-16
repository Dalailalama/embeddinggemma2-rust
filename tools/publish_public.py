"""Publish the current `main` of the development repository to the public GitHub repository.

usage: python tools/publish_public.py "what changed" [--tag v0.2.0] [--dry-run]

The public repository has its own history: each publication is one commit whose tree is exactly the
local `main`, authored with the GitHub noreply address, so neither the development history nor its
commit metadata (author e-mail addresses, intermediate commits) is published. `--tag` also pushes a
version tag, which makes the release workflow build binaries and mirror them to Hugging Face.
"""
import os
import subprocess
import sys

PUBLIC = "https://github.com/Dalailalama/embeddinggemma2-rust.git"
NAME, EMAIL = "Dalailalama", "158473629+Dalailalama@users.noreply.github.com"


def git(*args, env=None, check=True):
    p = subprocess.run(["git", *args], capture_output=True, text=True, env=env)
    if check and p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{p.stderr}")
    return p.stdout.strip()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    if tag in args:
        args.remove(tag)
    if len(args) != 1:
        raise SystemExit(__doc__)
    message = args[0]

    tree = git("rev-parse", "main^{tree}")
    # The public main, if it exists yet (the first publication has no parent).
    has_parent = subprocess.run(["git", "fetch", "--quiet", PUBLIC, "main"], capture_output=True).returncode == 0
    parent = git("rev-parse", "FETCH_HEAD") if has_parent else None
    if parent and git("rev-parse", f"{parent}^{{tree}}") == tree and not tag:
        print("public main already matches local main: nothing to publish")
        return
    env = dict(os.environ, GIT_AUTHOR_NAME=NAME, GIT_AUTHOR_EMAIL=EMAIL, GIT_COMMITTER_NAME=NAME, GIT_COMMITTER_EMAIL=EMAIL)
    if parent and git("rev-parse", f"{parent}^{{tree}}") == tree:
        commit = parent  # only a tag to add
    else:
        commit = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", message, env=env)
    print(f"snapshot {commit[:10]} of main ({'on top of ' + parent[:10] if parent else 'first publication'})")
    refs = [f"{commit}:refs/heads/main"] + ([f"{commit}:refs/tags/{tag}"] if tag else [])
    if dry:
        print("dry run: would push", " ".join(refs), "to", PUBLIC)
        return
    print(git("push", PUBLIC, *refs) or "pushed")


if __name__ == "__main__":
    main()
