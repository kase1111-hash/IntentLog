"""
Tests for Human-in-the-Loop (HITL) Triggers

Tests the HITL trigger infrastructure for requiring human approval
before executing sensitive operations.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from intentlog.triggers import (
    TriggerType,
    TriggerResponse,
    SensitivityLevel,
    TriggerRequest,
    TriggerResult,
    TriggerDeniedError,
    TriggerTimeoutError,
    TriggerHandlerBase,
    ConsoleTriggerHandler,
    CallbackTriggerHandler,
    set_trigger_handler,
    get_trigger_handler,
    get_trigger_history,
    clear_trigger_history,
    require_human_approval,
    notify_human,
    requires_approval,
    requires_confirmation,
    requires_review,
    TriggerScope,
    sensitive_operation,
    CommonTriggers,
)
from intentlog.context import intent_scope


class TestTriggerEnums:
    """Tests for trigger enums"""

    def test_trigger_types_exist(self):
        """Test that all trigger types exist"""
        assert TriggerType.NOTIFICATION.value == "notification"
        assert TriggerType.CONFIRMATION.value == "confirmation"
        assert TriggerType.APPROVAL.value == "approval"
        assert TriggerType.REVIEW.value == "review"

    def test_trigger_responses_exist(self):
        """Test that all trigger responses exist"""
        assert TriggerResponse.APPROVED.value == "approved"
        assert TriggerResponse.DENIED.value == "denied"
        assert TriggerResponse.MODIFIED.value == "modified"
        assert TriggerResponse.TIMEOUT.value == "timeout"
        assert TriggerResponse.ESCALATED.value == "escalated"
        assert TriggerResponse.SKIPPED.value == "skipped"

    def test_sensitivity_levels_exist(self):
        """Test that all sensitivity levels exist"""
        assert SensitivityLevel.LOW.value == "low"
        assert SensitivityLevel.MEDIUM.value == "medium"
        assert SensitivityLevel.HIGH.value == "high"
        assert SensitivityLevel.CRITICAL.value == "critical"


class TestTriggerRequest:
    """Tests for TriggerRequest"""

    def test_request_creation(self):
        """Test creating a trigger request"""
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.APPROVAL,
            operation_name="Delete Records",
            description="Delete 100 user records",
            sensitivity=SensitivityLevel.HIGH,
        )
        assert request.trigger_id == "test-123"
        assert request.trigger_type == TriggerType.APPROVAL
        assert request.operation_name == "Delete Records"
        assert request.sensitivity == SensitivityLevel.HIGH

    def test_request_with_intent_context(self):
        """Test request captures intent context"""
        with intent_scope("test-intent") as ctx:
            request = TriggerRequest(
                trigger_id="test-123",
                trigger_type=TriggerType.CONFIRMATION,
                operation_name="Test Op",
                description="Test",
                intent_context=ctx,
            )
            assert request.intent_id == ctx.intent_id
            assert request.intent_name == "test-intent"

    def test_request_to_dict(self):
        """Test serializing request to dict"""
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.APPROVAL,
            operation_name="Test Op",
            description="Test description",
            sensitivity=SensitivityLevel.CRITICAL,
            metadata={"key": "value"},
        )
        d = request.to_dict()
        assert d["trigger_id"] == "test-123"
        assert d["trigger_type"] == "approval"
        assert d["sensitivity"] == "critical"
        assert d["metadata"]["key"] == "value"


class TestTriggerResult:
    """Tests for TriggerResult"""

    def test_result_creation(self):
        """Test creating a trigger result"""
        result = TriggerResult(
            trigger_id="test-123",
            response=TriggerResponse.APPROVED,
            responded_by="test-user",
        )
        assert result.trigger_id == "test-123"
        assert result.response == TriggerResponse.APPROVED
        assert result.is_approved is True
        assert result.is_denied is False

    def test_result_denied(self):
        """Test denied result"""
        result = TriggerResult(
            trigger_id="test-123",
            response=TriggerResponse.DENIED,
            reason="Too risky",
        )
        assert result.is_approved is False
        assert result.is_denied is True
        assert result.reason == "Too risky"

    def test_result_modified(self):
        """Test modified result"""
        result = TriggerResult(
            trigger_id="test-123",
            response=TriggerResponse.MODIFIED,
            modifications={"amount": 50},
        )
        assert result.is_approved is True
        assert result.modifications == {"amount": 50}

    def test_result_timeout(self):
        """Test timeout counts as denied"""
        result = TriggerResult(
            trigger_id="test-123",
            response=TriggerResponse.TIMEOUT,
        )
        assert result.is_denied is True

    def test_result_to_dict(self):
        """Test serializing result to dict"""
        result = TriggerResult(
            trigger_id="test-123",
            response=TriggerResponse.APPROVED,
            responded_by="user1",
            reason="Looks good",
        )
        d = result.to_dict()
        assert d["trigger_id"] == "test-123"
        assert d["response"] == "approved"
        assert d["responded_by"] == "user1"
        assert d["reason"] == "Looks good"


class TestConsoleTriggerHandler:
    """Tests for ConsoleTriggerHandler"""

    def test_auto_approve_mode(self):
        """Test auto-approve mode for testing"""
        handler = ConsoleTriggerHandler(auto_approve=True)
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.APPROVAL,
            operation_name="Test Op",
            description="Test",
        )
        result = handler.handle(request)
        assert result.response == TriggerResponse.APPROVED
        assert result.responded_by == "auto"

    def test_notification_skipped(self):
        """Test that notifications are skipped in auto mode"""
        handler = ConsoleTriggerHandler(auto_approve=True)
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.NOTIFICATION,
            operation_name="Info",
            description="Just FYI",
        )
        result = handler.handle(request)
        assert result.response == TriggerResponse.APPROVED  # Auto-approved


class TestCallbackTriggerHandler:
    """Tests for CallbackTriggerHandler"""

    def test_callback_handler_approval(self):
        """Test callback handler for approval"""
        def approve_all(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.APPROVED,
            )

        handler = CallbackTriggerHandler(on_trigger=approve_all)
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.APPROVAL,
            operation_name="Test",
            description="Test",
        )
        result = handler.handle(request)
        assert result.response == TriggerResponse.APPROVED

    def test_callback_handler_denial(self):
        """Test callback handler for denial"""
        def deny_all(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.DENIED,
                reason="Policy violation",
            )

        handler = CallbackTriggerHandler(on_trigger=deny_all)
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.APPROVAL,
            operation_name="Test",
            description="Test",
        )
        result = handler.handle(request)
        assert result.response == TriggerResponse.DENIED
        assert result.reason == "Policy violation"

    def test_callback_notification(self):
        """Test callback handler for notifications"""
        notifications_received = []

        def on_notification(request):
            notifications_received.append(request.operation_name)

        handler = CallbackTriggerHandler(on_notification=on_notification)
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.NOTIFICATION,
            operation_name="Info Message",
            description="Test",
        )
        result = handler.handle(request)
        assert result.response == TriggerResponse.SKIPPED
        assert "Info Message" in notifications_received

    def test_no_handler_auto_approves(self):
        """Test that no handler results in auto-approval with warning"""
        handler = CallbackTriggerHandler()
        request = TriggerRequest(
            trigger_id="test-123",
            trigger_type=TriggerType.APPROVAL,
            operation_name="Test",
            description="Test",
        )
        result = handler.handle(request)
        assert result.response == TriggerResponse.APPROVED
        assert "auto-approved" in result.reason


class TestTriggerHandlerRegistry:
    """Tests for trigger handler registry"""

    def setup_method(self):
        """Reset handler before each test"""
        # Set up auto-approve handler for testing
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_set_and_get_handler(self):
        """Test setting and getting handler"""
        handler = ConsoleTriggerHandler(auto_approve=True)
        set_trigger_handler(handler)
        assert get_trigger_handler() is handler

    def test_default_handler_created(self):
        """Test that default handler is created if none set"""
        from intentlog import triggers
        triggers._default_handler = None
        handler = get_trigger_handler()
        assert isinstance(handler, ConsoleTriggerHandler)


class TestRequireHumanApproval:
    """Tests for require_human_approval function"""

    def setup_method(self):
        """Set up auto-approve handler for testing"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_basic_approval(self):
        """Test basic approval flow"""
        result = require_human_approval(
            operation_name="Test Op",
            description="Test operation",
        )
        assert result.is_approved

    def test_approval_with_sensitivity(self):
        """Test approval with sensitivity level"""
        result = require_human_approval(
            operation_name="Critical Op",
            description="Critical operation",
            sensitivity=SensitivityLevel.CRITICAL,
        )
        assert result.is_approved

    def test_approval_with_metadata(self):
        """Test approval includes metadata"""
        result = require_human_approval(
            operation_name="Test Op",
            description="Test",
            metadata={"count": 100},
        )
        assert result.is_approved

    def test_denial_raises_error(self):
        """Test that denial raises TriggerDeniedError"""
        def deny_all(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.DENIED,
                reason="Policy",
            )

        set_trigger_handler(CallbackTriggerHandler(on_trigger=deny_all))

        with pytest.raises(TriggerDeniedError) as exc_info:
            require_human_approval(
                operation_name="Test",
                description="Test",
            )
        assert "Policy" in str(exc_info.value)

    def test_denial_no_raise(self):
        """Test denial without raising error"""
        def deny_all(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.DENIED,
            )

        set_trigger_handler(CallbackTriggerHandler(on_trigger=deny_all))

        result = require_human_approval(
            operation_name="Test",
            description="Test",
            raise_on_deny=False,
        )
        assert result.is_denied

    def test_history_recorded(self):
        """Test that trigger history is recorded"""
        require_human_approval(
            operation_name="Test Op",
            description="Test",
        )
        history = get_trigger_history()
        assert len(history) == 1
        assert history[0]["request"]["operation_name"] == "Test Op"


class TestNotifyHuman:
    """Tests for notify_human function"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_notification_does_not_block(self):
        """Test that notification doesn't raise on 'denial'"""
        notify_human(
            operation_name="Info",
            description="Just information",
        )
        # Should not raise

    def test_notification_in_history(self):
        """Test that notification is recorded in history"""
        notify_human(
            operation_name="Status Update",
            description="Processing started",
        )
        history = get_trigger_history()
        assert len(history) == 1


class TestRequiresApprovalDecorator:
    """Tests for @requires_approval decorator"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_decorator_allows_execution(self):
        """Test decorated function executes when approved"""
        @requires_approval(description="Test function")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

    def test_decorator_blocks_on_denial(self):
        """Test decorated function blocked when denied"""
        def deny_all(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.DENIED,
            )

        set_trigger_handler(CallbackTriggerHandler(on_trigger=deny_all))

        @requires_approval(description="Test")
        def my_function():
            return "executed"

        with pytest.raises(TriggerDeniedError):
            my_function()

    def test_decorator_uses_function_name(self):
        """Test decorator uses function name as operation"""
        @requires_approval()
        def important_action():
            """Does something important"""
            pass

        important_action()
        history = get_trigger_history()
        assert history[0]["request"]["operation_name"] == "important_action"

    def test_decorator_uses_docstring(self):
        """Test decorator uses docstring as description"""
        @requires_approval()
        def do_something():
            """This is the description."""
            pass

        do_something()
        history = get_trigger_history()
        assert "description" in history[0]["request"]["description"]


class TestRequiresConfirmationDecorator:
    """Tests for @requires_confirmation decorator"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_confirmation_decorator(self):
        """Test confirmation decorator works"""
        @requires_confirmation("Are you sure?")
        def delete_something():
            return "deleted"

        result = delete_something()
        assert result == "deleted"


class TestRequiresReviewDecorator:
    """Tests for @requires_review decorator"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_review_decorator(self):
        """Test review decorator works"""
        @requires_review("Review this action")
        def review_action():
            return "reviewed"

        result = review_action()
        assert result == "reviewed"


class TestTriggerScope:
    """Tests for TriggerScope context manager"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_scope_approved(self):
        """Test scope when approved"""
        with TriggerScope(
            "Test Operation",
            "Test description",
        ) as scope:
            assert scope.is_approved
            # Code executes here

    def test_scope_denied_raises(self):
        """Test scope raises on denial"""
        def deny_all(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.DENIED,
            )

        set_trigger_handler(CallbackTriggerHandler(on_trigger=deny_all))

        with pytest.raises(TriggerDeniedError):
            with TriggerScope("Test", "Test"):
                pass

    def test_scope_denied_no_raise(self):
        """Test scope without raising on denial"""
        def deny_all(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.DENIED,
            )

        set_trigger_handler(CallbackTriggerHandler(on_trigger=deny_all))

        with TriggerScope("Test", "Test", raise_on_deny=False) as scope:
            assert not scope.is_approved

    def test_scope_modifications(self):
        """Test scope captures modifications"""
        def modify(request):
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.MODIFIED,
                modifications={"amount": 75},
            )

        set_trigger_handler(CallbackTriggerHandler(on_trigger=modify))

        with TriggerScope("Test", "Test") as scope:
            assert scope.is_approved
            assert scope.modifications == {"amount": 75}


class TestSensitiveOperation:
    """Tests for sensitive_operation helper"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_sensitive_operation(self):
        """Test sensitive_operation helper"""
        with sensitive_operation("Delete", "Delete all records"):
            pass  # Operation executes

        history = get_trigger_history()
        assert len(history) == 1
        assert history[0]["request"]["sensitivity"] == "high"


class TestCommonTriggers:
    """Tests for CommonTriggers predefined triggers"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_database_write(self):
        """Test database write trigger"""
        result = CommonTriggers.database_write("users", "insert", 10)
        assert result.is_approved
        history = get_trigger_history()
        assert "users" in history[0]["request"]["metadata"]["table"]

    def test_database_delete_is_critical(self):
        """Test database delete is critical sensitivity"""
        CommonTriggers.database_write("users", "delete", 100)
        history = get_trigger_history()
        assert history[0]["request"]["sensitivity"] == "critical"

    def test_email_send(self):
        """Test email send trigger"""
        result = CommonTriggers.email_send(50, "Newsletter")
        assert result.is_approved
        history = get_trigger_history()
        assert history[0]["request"]["operation_name"] == "Send Email"

    def test_bulk_email_is_high_sensitivity(self):
        """Test bulk email is high sensitivity"""
        CommonTriggers.email_send(1000, "Mass Email", is_bulk=True)
        history = get_trigger_history()
        assert history[0]["request"]["sensitivity"] == "high"

    def test_external_api_call(self):
        """Test external API call trigger"""
        result = CommonTriggers.external_api_call("payment-api", "POST")
        # Should be notification, not blocking
        assert result.response in (TriggerResponse.APPROVED, TriggerResponse.SKIPPED)

    def test_file_operation(self):
        """Test file operation trigger"""
        result = CommonTriggers.file_operation("/tmp/data.json", "write")
        assert result.is_approved

    def test_file_delete_is_high(self):
        """Test file delete is high sensitivity"""
        CommonTriggers.file_operation("/tmp/data.json", "delete")
        history = get_trigger_history()
        assert history[0]["request"]["sensitivity"] == "high"

    def test_payment_processing(self):
        """Test payment processing trigger"""
        result = CommonTriggers.payment_processing(99.99)
        assert result.is_approved

    def test_large_payment_is_critical(self):
        """Test large payment is critical sensitivity"""
        CommonTriggers.payment_processing(5000.00)
        history = get_trigger_history()
        assert history[0]["request"]["sensitivity"] == "critical"


class TestTriggerHistory:
    """Tests for trigger history management"""

    def setup_method(self):
        """Clear history before each test"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_history_accumulates(self):
        """Test that history accumulates"""
        require_human_approval("Op 1", "Test 1")
        require_human_approval("Op 2", "Test 2")
        require_human_approval("Op 3", "Test 3")

        history = get_trigger_history()
        assert len(history) == 3

    def test_history_limit(self):
        """Test history respects limit parameter"""
        for i in range(10):
            require_human_approval(f"Op {i}", f"Test {i}")

        history = get_trigger_history(limit=3)
        assert len(history) == 3

    def test_clear_history(self):
        """Test clearing history"""
        require_human_approval("Op", "Test")
        assert len(get_trigger_history()) == 1

        clear_trigger_history()
        assert len(get_trigger_history()) == 0


class TestIntentContextIntegration:
    """Tests for integration with intent context"""

    def setup_method(self):
        """Set up auto-approve handler"""
        set_trigger_handler(ConsoleTriggerHandler(auto_approve=True))
        clear_trigger_history()

    def test_trigger_captures_intent(self):
        """Test that trigger captures current intent context"""
        with intent_scope("parent-operation") as ctx:
            require_human_approval("Sub-op", "Test")

        history = get_trigger_history()
        assert history[0]["request"]["intent_name"] == "parent-operation"
        assert history[0]["request"]["intent_id"] == ctx.intent_id

    def test_trigger_without_intent(self):
        """Test trigger works without intent context"""
        require_human_approval("Standalone", "Test")
        history = get_trigger_history()
        assert history[0]["request"]["intent_id"] is None


class TestTriggerErrors:
    """Tests for trigger error classes"""

    def test_trigger_denied_error(self):
        """Test TriggerDeniedError"""
        error = TriggerDeniedError("test-123", "Policy violation")
        assert error.trigger_id == "test-123"
        assert error.reason == "Policy violation"
        assert "test-123" in str(error)
        assert "Policy violation" in str(error)

    def test_trigger_timeout_error(self):
        """Test TriggerTimeoutError"""
        error = TriggerTimeoutError("test-123", 30.0)
        assert error.trigger_id == "test-123"
        assert error.timeout_seconds == 30.0
        assert "30" in str(error)
