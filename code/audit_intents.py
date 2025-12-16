import sys
import re

def audit_logs(file_path):
    with open(file_path, 'r') as f:
        logs = f.read()

    errors = []

    # 1. Check for "Hallucination Risk" (Empty Intent reasoning)
    if re.search(r"intent_reasoning: ''", logs):
        errors.append("❌ Empty reasoning found in logs.")

    # 2. Check for "Cost/Loop Risk" (Same intent repeated > 3 times)
    # This is a simple regex check; can be more advanced with JSON parsing
    intents = re.findall(r"intent_name: '(\w+)'", logs)
    for name in set(intents):
        if intents.count(name) > 3:
            errors.append(f"⚠️ Potential Loop: Intent '{name}' called {intents.count(name)} times.")

    if errors:
        print("\n".join(errors))
        sys.exit(1) # This fails the GitHub Action
    else:
        print("✅ IntentLog Audit Passed!")
        sys.exit(0)

if __name__ == "__main__":
    audit_logs(sys.argv[1])
