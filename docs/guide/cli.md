# CLI Reference

Complete reference for IntentLog command-line interface.

## Global Options

```bash
ilog --version  # Show version
ilog --help     # Show help
```

## Core Commands

### init

Initialize a new IntentLog project.

```bash
ilog init <project-name> [--force]
```

| Option | Description |
|--------|-------------|
| `--force, -f` | Reinitialize existing project |

Auto-detects the git repository and stores `git_root` in the project config.

### commit

Create an intent commit.

```bash
ilog commit <message> [options]
```

| Option | Description |
|--------|-------------|
| `--attach, -a` | Attach git-tracked files |
| `--sign, -s` | Sign with default key |
| `--key-password` | Password for encrypted key |

Automatically captures git context (branch, HEAD hash, staged files) as `metadata.git_context`.

Prints the new intent as `[<hash>] <branch>: <name>`. `<hash>` is the 12-character
intent hash, the same identifier `ilog log` and `ilog search` print, and it can be
passed to `ilog show`, `ilog tag` and `ilog link`. The hash covers the intent's
content *and* metadata, so it changes as soon as the intent is tagged or linked.
Scripts that need a stable reference should capture the intent ID instead:
`ilog show <hash>` prints it on the `ID:` line and `ilog log --json` exposes it
as `intent_id`.

### log

Show intent history.

```bash
ilog log [--limit N] [--branch NAME] [--git-commit HASH] [--json]
```

| Option | Description |
|--------|-------------|
| `--limit, -n` | Number of intents (default: 10) |
| `--branch, -b` | Specific branch |
| `--git-commit` | Filter by associated git commit hash |
| `--json` | Output as JSON for scripting |

### show

Display a single intent with full metadata.

```bash
ilog show <id> [--branch NAME] [--json]
```

| Option | Description |
|--------|-------------|
| `--branch, -b` | Branch to search (default: current) |
| `--json` | Output as JSON for scripting |

`<id>` is a prefix of the intent ID, of the intent hash printed by `commit`,
`log` and `search`, or of the `chain_hash` stored in the branch file under
`.intentlog/` (what older versions of `ilog commit` printed). Only the intent ID
is stable: both hashes change once the intent is tagged or linked.

### search

Search intent history.

```bash
ilog search <query> [options]
```

| Option | Description |
|--------|-------------|
| `--branch, -b` | Search specific branch |
| `--semantic, -s` | Use LLM semantic search |
| `--top, -t` | Number of results (default: 5) |

### status

Show project status with git info.

```bash
ilog status [--json]
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON for scripting |

### branch

Manage branches.

```bash
ilog branch [name] [--list]
```

| Option | Description |
|--------|-------------|
| `--list, -l` | List all branches |

### diff

Show semantic diff between branches.

```bash
ilog diff <branch-spec>
```

Examples:
```bash
ilog diff feature         # main..feature
ilog diff main..feature   # explicit
```

### merge

Merge branches.

```bash
ilog merge <source> [--message MSG]
```

### blame

Show intent reasoning alongside git history for a file.

```bash
ilog blame <file>
```

Maps git log entries to associated intents, showing the reasoning behind each change.

### hooks

Manage git hooks for IntentLog.

```bash
ilog hooks <action>
```

Actions: `install`, `uninstall`, `status`

Installs a `prepare-commit-msg` git hook that prompts for intent when committing.

### config

Configure settings.

```bash
ilog config <setting> [options]
```

Settings:
- `llm` - Configure LLM provider
- `show` - Show current config

### audit

Audit intent logs.

```bash
ilog audit
```

### tag

Add or remove tags on an intent.

```bash
ilog tag <intent-id> <tag>... [--remove] [--branch NAME]
```

| Option | Description |
|--------|-------------|
| `--remove, -r` | Remove the tags instead of adding them |
| `--branch, -b` | Branch to search (default: current) |

### link

Link two related intents.

```bash
ilog link <source-id> <target-id> [--branch NAME]
```

| Option | Description |
|--------|-------------|
| `--branch, -b` | Branch holding the source intent (default: current) |

The target is looked up on the source's branch first, then on every other
branch, so a decision on `main` can link to an exploration recorded on a
feature branch. The link is stored on the source intent as `metadata.links`
(a list of target intent IDs) and listed by `ilog show`.

## Analytics Commands

### export

Export intents.

```bash
ilog export [options]
```

| Option | Description |
|--------|-------------|
| `--format, -f` | json, jsonl, csv, huggingface, openai |
| `--output, -o` | Output file |
| `--anonymize, -a` | Anonymize data |
| `--branch, -b` | Export from branch |
| `--start` | Filter start date (ISO) |
| `--end` | Filter end date (ISO) |

### analytics

Generate analytics.

```bash
ilog analytics [action] [options]
```

Actions: `summary`, `latency`, `frequency`, `errors`, `trends`, `bottlenecks`, `report`

### metrics

Compute doctrine metrics.

```bash
ilog metrics [action] [--branch BRANCH]
```

Actions: `all`, `density`, `info`, `auditability`, `fraud`

## Crypto Commands

### keys

Manage signing keys.

```bash
ilog keys <action> [options]
```

Actions: `generate`, `list`, `export`, `default`

| Option | Description |
|--------|-------------|
| `--name, -n` | Key name |
| `--password, -p` | Encryption password |
| `--output, -o` | Export file |

### chain

Manage intent chain.

```bash
ilog chain <action> [options]
```

Actions: `verify`, `migrate`, `status`, `proof`

| Option | Description |
|--------|-------------|
| `--branch, -b` | Target branch |
| `--sequence, -s` | Sequence for proof |

## Formalization Commands

### formalize

Derive formal outputs from prose (requires LLM configuration).

```bash
ilog formalize <action> [options]
```

Actions: `intent`, `chain`, `search`

| Option | Description |
|--------|-------------|
| `--type, -t` | code, rules, heuristics, schema, config, spec, tests |
| `--language, -l` | Programming language |
| `--intent-id, -i` | Target intent ID |
| `--query, -q` | Search query |
| `--output, -o` | Output file |

## Shell Completion

Generate shell completion scripts.

```bash
ilog completion [bash|zsh|fish]
```
