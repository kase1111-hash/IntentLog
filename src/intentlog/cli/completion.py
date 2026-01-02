"""
Shell Completion Scripts for IntentLog CLI

Provides shell completion scripts for bash, zsh, and fish shells.
"""

import sys
from typing import Optional

BASH_COMPLETION = '''# IntentLog bash completion
# Add to ~/.bashrc: source <(intentlog completion bash)
# Or save to /etc/bash_completion.d/intentlog

_intentlog_completions() {
    local cur prev words cword
    _init_completion || return

    local commands="init commit log search branch status diff merge config audit
                    observe segment receipt ledger verify
                    export analytics metrics sufficiency
                    keys chain
                    privacy
                    formalize
                    backup completion"

    local init_opts="--force"
    local commit_opts="--parent --attach --sign --no-chain"
    local log_opts="--limit --format --branch --from-date --to-date"
    local search_opts="--branch --limit"
    local branch_opts="--list --delete --from"
    local export_opts="--format --output --filter --from-date --to-date --anonymize"
    local backup_opts="--list --restore --verify --delete --cleanup"

    case $prev in
        intentlog|ilog)
            COMPREPLY=($(compgen -W "$commands" -- "$cur"))
            return 0
            ;;
        init)
            COMPREPLY=($(compgen -W "$init_opts" -- "$cur"))
            return 0
            ;;
        commit)
            COMPREPLY=($(compgen -W "$commit_opts" -- "$cur"))
            return 0
            ;;
        log)
            COMPREPLY=($(compgen -W "$log_opts" -- "$cur"))
            return 0
            ;;
        search)
            COMPREPLY=($(compgen -W "$search_opts" -- "$cur"))
            return 0
            ;;
        branch)
            COMPREPLY=($(compgen -W "$branch_opts" -- "$cur"))
            return 0
            ;;
        export)
            COMPREPLY=($(compgen -W "$export_opts" -- "$cur"))
            return 0
            ;;
        --format)
            COMPREPLY=($(compgen -W "json jsonl csv huggingface openai" -- "$cur"))
            return 0
            ;;
        --branch|--from)
            # Complete branch names
            local branches=$(intentlog branch --list 2>/dev/null | tr '\n' ' ')
            COMPREPLY=($(compgen -W "$branches" -- "$cur"))
            return 0
            ;;
        backup)
            COMPREPLY=($(compgen -W "$backup_opts" -- "$cur"))
            return 0
            ;;
        completion)
            COMPREPLY=($(compgen -W "bash zsh fish" -- "$cur"))
            return 0
            ;;
    esac

    COMPREPLY=($(compgen -W "$commands" -- "$cur"))
}

complete -F _intentlog_completions intentlog
complete -F _intentlog_completions ilog
'''

ZSH_COMPLETION = '''#compdef intentlog ilog
# IntentLog zsh completion
# Add to ~/.zshrc: source <(intentlog completion zsh)
# Or save to ~/.zsh/completions/_intentlog

_intentlog() {
    local -a commands
    commands=(
        'init:Initialize a new IntentLog project'
        'commit:Record a new intent'
        'log:View intent history'
        'search:Search intents'
        'branch:Manage branches'
        'status:Show project status'
        'diff:Compare branches'
        'merge:Merge branches'
        'config:Manage configuration'
        'audit:View audit log'
        'observe:Start observation session'
        'segment:Create time segment'
        'receipt:Generate effort receipt'
        'ledger:Manage receipt ledger'
        'verify:Verify receipt'
        'export:Export intents'
        'analytics:View analytics'
        'metrics:Compute metrics'
        'sufficiency:Test intent sufficiency'
        'keys:Manage signing keys'
        'chain:Verify chain integrity'
        'privacy:Manage privacy settings'
        'formalize:Formalize intents'
        'backup:Backup and restore'
        'completion:Generate shell completion'
    )

    local -a formats
    formats=(
        'json:JSON format'
        'jsonl:JSON Lines format'
        'csv:CSV format'
        'huggingface:HuggingFace format'
        'openai:OpenAI format'
    )

    _arguments -C \\
        '--version[Show version]' \\
        '--help[Show help]' \\
        '1: :->command' \\
        '*:: :->args'

    case $state in
        command)
            _describe -t commands 'intentlog commands' commands
            ;;
        args)
            case $words[1] in
                init)
                    _arguments \\
                        '--force[Reinitialize existing project]' \\
                        ':project_name:'
                    ;;
                commit)
                    _arguments \\
                        '--parent[Parent intent ID]:parent_id:' \\
                        '--attach[Attach files]' \\
                        '--sign[Sign the intent]' \\
                        '--no-chain[Skip chain linking]' \\
                        ':name:' \\
                        ':reasoning:'
                    ;;
                log)
                    _arguments \\
                        '--limit[Number of entries]:limit:' \\
                        '--format[Output format]:format:(short full json)' \\
                        '--branch[Branch name]:branch:' \\
                        '--from-date[Start date]:date:' \\
                        '--to-date[End date]:date:'
                    ;;
                search)
                    _arguments \\
                        '--branch[Branch to search]:branch:' \\
                        '--limit[Max results]:limit:' \\
                        ':query:'
                    ;;
                branch)
                    _arguments \\
                        '--list[List branches]' \\
                        '--delete[Delete branch]' \\
                        '--from[Source branch]:branch:' \\
                        ':branch_name:'
                    ;;
                export)
                    _arguments \\
                        '--format[Output format]:format:((${(j: :)${(@qq)formats}}))'\\
                        '--output[Output file]:file:_files' \\
                        '--filter[Filter expression]:filter:' \\
                        '--anonymize[Anonymize output]'
                    ;;
                backup)
                    _arguments \\
                        '--list[List backups]' \\
                        '--restore[Restore backup]:backup_id:' \\
                        '--verify[Verify backup]:backup_id:' \\
                        '--delete[Delete backup]:backup_id:' \\
                        '--cleanup[Cleanup old backups]'
                    ;;
                completion)
                    _arguments ':shell:(bash zsh fish)'
                    ;;
            esac
            ;;
    esac
}

compdef _intentlog intentlog
compdef _intentlog ilog
'''

FISH_COMPLETION = '''# IntentLog fish completion
# Save to ~/.config/fish/completions/intentlog.fish

set -l commands init commit log search branch status diff merge config audit \\
    observe segment receipt ledger verify \\
    export analytics metrics sufficiency \\
    keys chain privacy formalize backup completion

# Disable file completion by default
complete -c intentlog -f
complete -c ilog -f

# Main commands
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a init -d "Initialize project"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a commit -d "Record intent"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a log -d "View history"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a search -d "Search intents"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a branch -d "Manage branches"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a status -d "Show status"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a diff -d "Compare branches"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a merge -d "Merge branches"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a config -d "Manage config"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a audit -d "View audit log"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a observe -d "Start observation"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a segment -d "Create segment"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a receipt -d "Generate receipt"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a ledger -d "Manage ledger"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a verify -d "Verify receipt"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a export -d "Export intents"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a analytics -d "View analytics"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a metrics -d "Compute metrics"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a sufficiency -d "Test sufficiency"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a keys -d "Manage keys"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a chain -d "Verify chain"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a privacy -d "Privacy settings"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a formalize -d "Formalize intents"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a backup -d "Backup/restore"
complete -c intentlog -n "not __fish_seen_subcommand_from $commands" -a completion -d "Shell completion"

# Copy completions for ilog alias
complete -c ilog -w intentlog

# init options
complete -c intentlog -n "__fish_seen_subcommand_from init" -l force -d "Reinitialize"

# commit options
complete -c intentlog -n "__fish_seen_subcommand_from commit" -l parent -d "Parent ID"
complete -c intentlog -n "__fish_seen_subcommand_from commit" -l attach -d "Attach files"
complete -c intentlog -n "__fish_seen_subcommand_from commit" -l sign -d "Sign intent"
complete -c intentlog -n "__fish_seen_subcommand_from commit" -l no-chain -d "Skip chain"

# log options
complete -c intentlog -n "__fish_seen_subcommand_from log" -l limit -d "Max entries"
complete -c intentlog -n "__fish_seen_subcommand_from log" -l format -a "short full json" -d "Output format"
complete -c intentlog -n "__fish_seen_subcommand_from log" -l branch -d "Branch name"

# export options
complete -c intentlog -n "__fish_seen_subcommand_from export" -l format -a "json jsonl csv huggingface openai" -d "Format"
complete -c intentlog -n "__fish_seen_subcommand_from export" -l output -r -d "Output file"
complete -c intentlog -n "__fish_seen_subcommand_from export" -l anonymize -d "Anonymize"

# backup options
complete -c intentlog -n "__fish_seen_subcommand_from backup" -l list -d "List backups"
complete -c intentlog -n "__fish_seen_subcommand_from backup" -l restore -d "Restore backup"
complete -c intentlog -n "__fish_seen_subcommand_from backup" -l verify -d "Verify backup"
complete -c intentlog -n "__fish_seen_subcommand_from backup" -l delete -d "Delete backup"
complete -c intentlog -n "__fish_seen_subcommand_from backup" -l cleanup -d "Cleanup old"

# completion shells
complete -c intentlog -n "__fish_seen_subcommand_from completion" -a "bash zsh fish" -d "Shell"
'''


def get_completion_script(shell: str) -> Optional[str]:
    """
    Get the completion script for a shell.

    Args:
        shell: Shell name (bash, zsh, fish)

    Returns:
        Completion script string or None if unsupported
    """
    scripts = {
        "bash": BASH_COMPLETION,
        "zsh": ZSH_COMPLETION,
        "fish": FISH_COMPLETION,
    }
    return scripts.get(shell.lower())


def print_completion(shell: str) -> None:
    """Print completion script for a shell."""
    script = get_completion_script(shell)
    if script:
        print(script)
    else:
        print(f"Unsupported shell: {shell}", file=sys.stderr)
        print("Supported shells: bash, zsh, fish", file=sys.stderr)
        sys.exit(1)


def register_completion_commands(subparsers) -> None:
    """Register completion command."""
    completion_parser = subparsers.add_parser(
        "completion",
        help="Generate shell completion scripts"
    )
    completion_parser.add_argument(
        "shell",
        choices=["bash", "zsh", "fish"],
        help="Shell to generate completion for"
    )
    completion_parser.set_defaults(func=lambda args: print_completion(args.shell))


__all__ = [
    "get_completion_script",
    "print_completion",
    "register_completion_commands",
    "BASH_COMPLETION",
    "ZSH_COMPLETION",
    "FISH_COMPLETION",
]
