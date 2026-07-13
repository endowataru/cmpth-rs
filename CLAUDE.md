# CLAUDE.md — cmpth-rs

## Commit messages

Always use **Conventional Commits** (semantic commit messages):

```
<type>(<scope>): <short summary>
```

**Types:** `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`, `build`

**Rules:**
- Use present tense, imperative mood ("add", not "adds" or "added")
- Keep the summary line under 72 characters
- Do NOT use bare `Fix`, `Revert`, or bare imperative without a type prefix
- Do NOT include `Co-Authored-By:` or `Claude-Session:` trailers

**Examples:**
```
feat: add work-stealing scheduler
fix: correct AArch64 context-switch memory ordering
refactor: introduce UltContextSystem/UltSchedulerSystem trait layers
perf: reduce false sharing in MCS lock queue nodes
test: add barrier_sync stress test
docs: document UltContext safety invariants
```
