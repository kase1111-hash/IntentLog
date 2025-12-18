"""
Core IntentLog functionality

This module provides the fundamental data structures and operations for
managing intent logs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


@dataclass
class Intent:
    """Represents a single intent record"""

    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    intent_name: str = ""
    intent_reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_intent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary format"""
        return {
            "intent_id": self.intent_id,
            "intent_name": self.intent_name,
            "intent_reasoning": self.intent_reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "parent_intent_id": self.parent_intent_id,
        }

    def validate(self) -> bool:
        """Validate intent has required fields"""
        if not self.intent_name:
            return False
        if not self.intent_reasoning or self.intent_reasoning.strip() == "":
            return False
        return True


class IntentLog:
    """Main IntentLog manager class"""

    def __init__(self, project_name: str = ""):
        self.project_name = project_name
        self.intents: List[Intent] = []
        self.current_branch = "main"

    def add_intent(self, name: str, reasoning: str, metadata: Optional[Dict[str, Any]] = None,
                   parent_id: Optional[str] = None) -> Intent:
        """Add a new intent to the log"""
        intent = Intent(
            intent_name=name,
            intent_reasoning=reasoning,
            metadata=metadata or {},
            parent_intent_id=parent_id
        )

        if not intent.validate():
            raise ValueError("Intent must have name and non-empty reasoning")

        self.intents.append(intent)
        return intent

    def get_intent_chain(self, intent_id: str) -> List[Intent]:
        """Get the full chain of intents leading to this one"""
        chain = []
        current_id = intent_id

        while current_id:
            intent = self.get_intent(current_id)
            if not intent:
                break
            chain.insert(0, intent)
            current_id = intent.parent_intent_id

        return chain

    def get_intent(self, intent_id: str) -> Optional[Intent]:
        """Retrieve an intent by ID"""
        for intent in self.intents:
            if intent.intent_id == intent_id:
                return intent
        return None

    def search_intents(self, query: str) -> List[Intent]:
        """Search intents by name or reasoning content"""
        query_lower = query.lower()
        results = []

        for intent in self.intents:
            if (query_lower in intent.intent_name.lower() or
                query_lower in intent.intent_reasoning.lower()):
                results.append(intent)

        return results

    def export_to_dict(self) -> Dict[str, Any]:
        """Export the entire log to a dictionary"""
        return {
            "project_name": self.project_name,
            "current_branch": self.current_branch,
            "intents": [intent.to_dict() for intent in self.intents],
        }
