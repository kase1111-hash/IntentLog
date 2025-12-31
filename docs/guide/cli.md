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

### branch

Manage branches.

```bash
ilog branch [name] [--list]
```

| Option | Description |
|--------|-------------|
| `--list, -l` | List all branches |

### log

Show intent history.

```bash
ilog log [--limit N] [--branch NAME]
```

| Option | Description |
|--------|-------------|
| `--limit, -n` | Number of intents (default: 10) |
| `--branch, -b` | Specific branch |

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

Show project status.

```bash
ilog status
```

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

### config

Configure settings.

```bash
ilog config <setting> [options]
```

Settings:
- `llm` - Configure LLM provider
- `show` - Show current config

## MP-02 Commands

### observe

Manage observation sessions.

```bash
ilog observe <action> [paths...]
```

Actions: `start`, `stop`, `status`

### segment

Mark segment boundaries.

```bash
ilog segment <action> [text] [--category CAT]
```

Actions: `mark`, `list`

### receipt

Manage effort receipts.

```bash
ilog receipt <action> [receipt-id]
```

Actions: `create`, `list`, `show`, `verify`

### ledger

Manage append-only ledger.

```bash
ilog ledger <action> [options]
```

Actions: `show`, `verify`, `export`, `stats`, `checkpoint`

### verify

Verify integrity.

```bash
ilog verify [target]
```

Targets: `all`, `ledger`, `receipts`, `<receipt-id>`

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

### sufficiency

Run Intent Sufficiency Test.

```bash
ilog sufficiency [--branch BRANCH] [--author AUTHOR] [--verbose]
```

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

## Privacy Commands

### privacy

Manage privacy controls.

```bash
ilog privacy <action> [options]
```

Actions: `status`, `revoke`, `list`, `encrypt`, `keys`

| Option | Description |
|--------|-------------|
| `--target, -t` | all, intent, session |
| `--target-id` | ID to revoke |
| `--reason, -r` | Revocation reason |
| `--user-id, -u` | User performing action |

## Formalization Commands

### formalize

Derive formal outputs from prose.

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
