"""
Formalization CLI commands for IntentLog.

Commands: formalize
"""

import sys
from pathlib import Path

from ..storage import IntentLogStorage, ProjectNotFoundError, BranchNotFoundError
from ..semantic import FormalizationType, SemanticEngine
from .utils import load_config_or_exit


def cmd_formalize(args):
    """Derive formal code/rules/heuristics from prose intent"""
    action = args.action
    output_type = getattr(args, 'type', 'code')
    language = getattr(args, 'language', None)
    intent_id = getattr(args, 'intent_id', None)
    query = getattr(args, 'query', None)
    branch = getattr(args, 'branch', None)
    output_file = getattr(args, 'output', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
        intents = storage.load_intents(branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not intents:
        print(f"No intents on branch '{branch}'")
        return

    # Map string to FormalizationType
    type_map = {
        'code': FormalizationType.CODE,
        'rules': FormalizationType.RULES,
        'heuristics': FormalizationType.HEURISTICS,
        'schema': FormalizationType.SCHEMA,
        'config': FormalizationType.CONFIG,
        'spec': FormalizationType.SPEC,
        'tests': FormalizationType.TESTS,
    }
    formalization_type = type_map.get(output_type, FormalizationType.CODE)

    # Get LLM provider
    from ..llm.registry import get_provider

    llm_config = config.llm_config or {}
    provider_name = llm_config.get("provider", "mock")

    try:
        provider = get_provider(provider_name, llm_config)
    except Exception as e:
        print(f"Error: Could not initialize LLM provider: {e}")
        print("Configure LLM with: ilog config llm --provider openai")
        sys.exit(1)

    engine = SemanticEngine(provider)

    try:
        if action == "intent":
            # Formalize a specific intent by ID
            if not intent_id:
                print("Error: --intent-id required for 'intent' action")
                sys.exit(1)

            # Find intent by ID (partial match)
            target_intent = None
            for intent in intents:
                if intent.intent_id.startswith(intent_id):
                    target_intent = intent
                    break

            if not target_intent:
                print(f"Error: No intent found matching ID '{intent_id}'")
                sys.exit(1)

            print(f"Formalizing intent: {target_intent.intent_name}")
            print(f"  ID: {target_intent.intent_id[:12]}...")
            print(f"  Type: {output_type.upper()}")
            if language:
                print(f"  Language: {language}")
            print()

            result = engine.formalize(
                target_intent,
                formalization_type=formalization_type,
                language=language,
            )

        elif action == "chain":
            # Formalize from the entire chain of intents
            print(f"Formalizing chain of {len(intents)} intents")
            print(f"  Type: {output_type.upper()}")
            if language:
                print(f"  Language: {language}")
            print()

            result = engine.formalize_chain(
                intents,
                formalization_type=formalization_type,
                language=language,
            )

        elif action == "search":
            # Formalize from search results
            if not query:
                print("Error: --query required for 'search' action")
                sys.exit(1)

            print(f"Searching for intents matching: '{query}'")
            print(f"  Type: {output_type.upper()}")
            if language:
                print(f"  Language: {language}")
            print()

            result = engine.formalize_from_search(
                query,
                intents,
                formalization_type=formalization_type,
                language=language,
            )

        else:
            print(f"Unknown action: {action}")
            sys.exit(1)

        # Display result
        print("=" * 60)
        print(f"FORMALIZED OUTPUT ({output_type.upper()})")
        print("=" * 60)

        if result.language:
            print(f"Language: {result.language}")
        print(f"Confidence: {result.confidence:.0%}")
        print()
        print(result.content)
        print()

        if result.explanation:
            print("-" * 60)
            print("EXPLANATION:")
            print(result.explanation)
            print()

        if result.warnings:
            print("-" * 60)
            print("WARNINGS:")
            for warning in result.warnings:
                print(f"  - {warning}")
            print()

        print("-" * 60)
        print("PROVENANCE:")
        print(f"  Source intents: {len(result.provenance.source_intent_ids)}")
        for sid in result.provenance.source_intent_ids[:5]:
            print(f"    - {sid[:12]}...")
        if len(result.provenance.source_intent_ids) > 5:
            print(f"    ... and {len(result.provenance.source_intent_ids) - 5} more")
        print(f"  Formalized at: {result.provenance.formalized_at}")
        print(f"  Model: {result.provenance.model}")
        print("=" * 60)

        # Output to file if requested
        if output_file:
            import json
            output_path = Path(output_file)
            output_path.write_text(json.dumps(result.to_dict(), indent=2))
            print(f"\nSaved to: {output_file}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during formalization: {e}")
        sys.exit(1)


def register_formalize_commands(subparsers):
    """Register formalize commands with the argument parser."""
    # formalize command (Deferred Formalization)
    formalize_parser = subparsers.add_parser("formalize", help="Derive code/rules/heuristics from prose intent")
    formalize_parser.add_argument("action", choices=["intent", "chain", "search"],
                                  help="Formalization action: intent (single), chain (all), search (by query)")
    formalize_parser.add_argument("--type", "-t", default="code",
                                  choices=["code", "rules", "heuristics", "schema", "config", "spec", "tests"],
                                  help="Output type (default: code)")
    formalize_parser.add_argument("--language", "-l",
                                  help="Programming language for code output (default: python)")
    formalize_parser.add_argument("--intent-id", "-i",
                                  help="Intent ID to formalize (for 'intent' action)")
    formalize_parser.add_argument("--query", "-q",
                                  help="Search query (for 'search' action)")
    formalize_parser.add_argument("--branch", "-b", help="Branch to formalize from")
    formalize_parser.add_argument("--output", "-o", help="Output file (saves full result as JSON)")
    formalize_parser.set_defaults(func=cmd_formalize)
