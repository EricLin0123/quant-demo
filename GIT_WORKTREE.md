# Git Worktree Guide — quant-demo

This guide covers working with two simultaneous checkouts of this repo:

| Worktree | Path | Branch | Purpose |
|---|---|---|---|
| primary | `~/quant-demo` | `main` | current development |
| secondary | `~/quant-demo-long-short` | `long-short` | feature at commit `caa1865a` |

---

## 1. Initial Setup

Run this **once** from the primary directory:

```bash
git worktree add -b long-short ../quant-demo-long-short caa1865a
```

This creates a new directory `../quant-demo-long-short` checked out at `caa1865a`, and simultaneously creates a local branch named `long-short` pointing there. Both directories share the same `.git` folder — no data is duplicated.

---

## 2. Directory Layout

```
~/
├── quant-demo/               ← primary worktree  (branch: main)
│   └── .git/                 ← shared git database
│       └── worktrees/
│           └── quant-demo-long-short/
└── quant-demo-long-short/    ← secondary worktree (branch: long-short)
```

Each worktree has its own:
- working tree (files on disk)
- `HEAD`, `MERGE_HEAD`, staging area

They share everything inside `.git/objects` (history, commits, blobs).

---

## 3. Switching Between Worktrees

There is no `git checkout` needed. Just `cd`:

```bash
# go to the feature
cd ~/quant-demo-long-short

# go back to main
cd ~/quant-demo
# or, from anywhere:
cd -    # toggles to the previous directory
```

Your shell's working directory is your "active" worktree. Git commands run relative to where you are.

---

## 4. Listing Worktrees

```bash
git worktree list
```

Example output:
```
/Users/wanchuan/quant-demo               58fef13 [main]
/Users/wanchuan/quant-demo-long-short    caa1865a [long-short]
```

---

## 5. Committing in Each Worktree

Normal git workflow, scoped to the directory you're in:

```bash
# in ~/quant-demo-long-short
git add src/some_file.py
git commit -m "tweak long-short feature"

# in ~/quant-demo
git add src/other_file.py
git commit -m "continue main work"
```

Each commit lands on the branch of that worktree. They don't interfere.

---

## 6. Running the Pipeline in Each Worktree

Both worktrees are independent Python environments. Run from the relevant directory:

```bash
# primary
cd ~/quant-demo
PYTHONPATH=src uv run python -c "from twii_forecast import pipeline; pipeline.run()"

# feature branch
cd ~/quant-demo-long-short
PYTHONPATH=src uv run python -c "from twii_forecast import pipeline; pipeline.run()"
```

If `uv` resolves the venv from `.venv/` inside each directory, you may need `uv sync` once in the secondary worktree.

---

## 7. Pulling Updates into long-short

If `long-short` has a remote tracking branch:

```bash
cd ~/quant-demo-long-short
git pull
```

Or from the primary directory without switching:

```bash
git -C ~/quant-demo-long-short pull
```

To rebase `long-short` on top of updated `main`:

```bash
cd ~/quant-demo-long-short
git rebase main
```

---

## 8. Merging long-short into main

From the primary worktree:

```bash
cd ~/quant-demo
git merge long-short
```

Or cherry-pick specific commits:

```bash
git cherry-pick <commit-hash>
```

---

## 9. Removing the Worktree When Done

```bash
# from anywhere
git worktree remove ~/quant-demo-long-short

# optionally delete the branch too
git branch -d long-short
```

If the worktree has uncommitted changes, add `--force`:

```bash
git worktree remove --force ~/quant-demo-long-short
```

---

## 10. Gotchas

| Situation | What happens | Fix |
|---|---|---|
| Check out `main` inside the long-short worktree | Git refuses — a branch can only be active in one worktree at a time | Use a different branch name, or remove the primary worktree first |
| Two git commands run simultaneously in both worktrees | One may get a `.git/index.lock` error | Wait for the first to finish; delete the lock file if the process died |
| `uv run` picks up the wrong venv | Commands may fail with import errors | Run `uv sync` inside the secondary worktree to create its own `.venv` |
| Forgot which directory you're in | `git worktree list` shows all active paths and their branches | — |
