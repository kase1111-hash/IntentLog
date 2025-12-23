"""
Tests for Phase 5: Decorator and Context Management

Tests for:
- Context management (context.py)
- @intent_logger decorator (decorator.py)
- Session tracking
- Nested intent tracing
"""

import pytest
import asyncio
import tempfile
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

from intentlog.context import (
    IntentContext,
    SessionContext,
    IntentContextManager,
    SessionContextManager,
    get_current_intent,
    set_current_intent,
    get_current_session,
    set_current_session,
    intent_scope,
    session_scope,
    get_intent_chain,
    get_current_depth,
    get_session_id,
)

from intentlog.decorator import (
    intent_logger,
    intent_logger_class,
    IntentLoggerConfig,
    LogLevel,
    set_log_level,
    get_log_level,
    should_log,
    get_intent_log,
    clear_intent_log,
)


# ============================================================================
# Context Tests
# ============================================================================

class TestIntentContext:
    """Tests for IntentContext dataclass"""

    def test_create_context(self):
        """Create basic context"""
        ctx = IntentContext(
            intent_id="test-123",
            intent_name="test-intent",
        )
        assert ctx.intent_id == "test-123"
        assert ctx.intent_name == "test-intent"
        assert ctx.depth == 0
        assert ctx.parent_id is None

    def test_context_depth(self):
        """Context depth tracks nesting"""
        parent = IntentContext(intent_id="parent", intent_name="parent")
        child = IntentContext(
            intent_id="child",
            intent_name="child",
            parent_context=parent,
        )
        grandchild = IntentContext(
            intent_id="grandchild",
            intent_name="grandchild",
            parent_context=child,
        )

        assert parent.depth == 0
        assert child.depth == 1
        assert grandchild.depth == 2

    def test_context_parent_id(self):
        """Parent ID is correctly derived"""
        parent = IntentContext(intent_id="parent-id", intent_name="parent")
        child = IntentContext(
            intent_id="child-id",
            intent_name="child",
            parent_context=parent,
        )

        assert child.parent_id == "parent-id"

    def test_context_elapsed_time(self):
        """Elapsed time is calculated"""
        ctx = IntentContext(intent_id="test", intent_name="test")
        time.sleep(0.01)  # 10ms
        assert ctx.elapsed_ms >= 10

    def test_context_to_dict(self):
        """Context can be serialized"""
        ctx = IntentContext(
            intent_id="test",
            intent_name="test",
            session_id="session-1",
            metadata={"key": "value"},
        )
        data = ctx.to_dict()

        assert data["intent_id"] == "test"
        assert data["intent_name"] == "test"
        assert data["session_id"] == "session-1"
        assert data["metadata"]["key"] == "value"


class TestSessionContext:
    """Tests for SessionContext dataclass"""

    def test_create_session(self):
        """Create basic session"""
        session = SessionContext(
            session_id="sess-123",
            user_id="user-456",
        )
        assert session.session_id == "sess-123"
        assert session.user_id == "user-456"
        assert session.intent_count == 0

    def test_session_auto_id(self):
        """Session auto-generates ID"""
        session = SessionContext()
        assert session.session_id is not None
        assert len(session.session_id) == 8


class TestContextManagement:
    """Tests for context get/set functions"""

    def setup_method(self):
        """Clear context before each test"""
        set_current_intent(None)
        set_current_session(None)

    def test_get_set_current_intent(self):
        """Get and set current intent"""
        assert get_current_intent() is None

        ctx = IntentContext(intent_id="test", intent_name="test")
        set_current_intent(ctx)

        assert get_current_intent() == ctx

    def test_get_set_current_session(self):
        """Get and set current session"""
        assert get_current_session() is None

        session = SessionContext()
        set_current_session(session)

        assert get_current_session() == session


class TestIntentContextManager:
    """Tests for IntentContextManager"""

    def setup_method(self):
        set_current_intent(None)
        set_current_session(None)

    def test_basic_context_manager(self):
        """Basic context manager usage"""
        with IntentContextManager("test-intent") as ctx:
            assert ctx.intent_name == "test-intent"
            assert get_current_intent() == ctx

        assert get_current_intent() is None

    def test_nested_context_managers(self):
        """Nested context managers track parent"""
        with IntentContextManager("parent") as parent_ctx:
            assert parent_ctx.depth == 0

            with IntentContextManager("child") as child_ctx:
                assert child_ctx.depth == 1
                assert child_ctx.parent_id == parent_ctx.intent_id
                assert get_current_intent() == child_ctx

            assert get_current_intent() == parent_ctx

    def test_context_with_session(self):
        """Context picks up session ID"""
        with SessionContextManager(session_id="sess-123"):
            with IntentContextManager("test") as ctx:
                assert ctx.session_id == "sess-123"

    def test_child_tracking(self):
        """Parent tracks children"""
        with IntentContextManager("parent") as parent:
            with IntentContextManager("child1") as child1:
                pass
            with IntentContextManager("child2") as child2:
                pass

            assert child1.intent_id in parent.children
            assert child2.intent_id in parent.children


class TestSessionContextManager:
    """Tests for SessionContextManager"""

    def setup_method(self):
        set_current_session(None)

    def test_basic_session_manager(self):
        """Basic session manager usage"""
        with SessionContextManager(user_id="user-123") as session:
            assert session.user_id == "user-123"
            assert get_current_session() == session

        assert get_current_session() is None

    def test_session_intent_count(self):
        """Session tracks intent count"""
        with SessionContextManager() as session:
            assert session.intent_count == 0

            with IntentContextManager("first"):
                pass
            assert session.intent_count == 1

            with IntentContextManager("second"):
                pass
            assert session.intent_count == 2


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def setup_method(self):
        set_current_intent(None)
        set_current_session(None)

    def test_intent_scope(self):
        """intent_scope convenience function"""
        with intent_scope("my-intent") as ctx:
            assert ctx.intent_name == "my-intent"

    def test_session_scope(self):
        """session_scope convenience function"""
        with session_scope(user_id="user-1") as session:
            assert session.user_id == "user-1"

    def test_get_intent_chain(self):
        """get_intent_chain returns full chain"""
        with intent_scope("root") as root:
            with intent_scope("middle") as middle:
                with intent_scope("leaf") as leaf:
                    chain = get_intent_chain()

                    assert len(chain) == 3
                    assert chain[0] == root
                    assert chain[1] == middle
                    assert chain[2] == leaf

    def test_get_current_depth(self):
        """get_current_depth tracks nesting (0-indexed, root=0)"""
        assert get_current_depth() == 0

        with intent_scope("level1"):
            assert get_current_depth() == 0  # First intent is root (depth 0)

            with intent_scope("level2"):
                assert get_current_depth() == 1  # Child of root (depth 1)

    def test_get_session_id(self):
        """get_session_id returns current session"""
        assert get_session_id() is None

        with session_scope(session_id="test-session"):
            assert get_session_id() == "test-session"


class TestAsyncContext:
    """Tests for async context management"""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Async context manager works"""
        set_current_intent(None)

        async with IntentContextManager("async-test") as ctx:
            assert ctx.intent_name == "async-test"
            assert get_current_intent() == ctx

        assert get_current_intent() is None

    @pytest.mark.asyncio
    async def test_async_nested_contexts(self):
        """Nested async contexts work"""
        set_current_intent(None)

        async with IntentContextManager("parent") as parent:
            async with IntentContextManager("child") as child:
                assert child.parent_id == parent.intent_id


# ============================================================================
# Decorator Tests
# ============================================================================

class TestLogLevel:
    """Tests for log level management"""

    def test_log_levels(self):
        """Log levels are ordered correctly"""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.IMPORTANT.value
        assert LogLevel.IMPORTANT.value < LogLevel.CRITICAL.value
        assert LogLevel.CRITICAL.value < LogLevel.OFF.value

    def test_should_log(self):
        """should_log respects log level"""
        set_log_level(LogLevel.INFO)

        assert should_log(LogLevel.INFO)
        assert should_log(LogLevel.IMPORTANT)
        assert should_log(LogLevel.CRITICAL)
        assert not should_log(LogLevel.DEBUG)

    def test_set_get_log_level(self):
        """Log level can be set and retrieved"""
        set_log_level(LogLevel.DEBUG)
        assert get_log_level() == LogLevel.DEBUG

        set_log_level(LogLevel.CRITICAL)
        assert get_log_level() == LogLevel.CRITICAL


class TestIntentLogger:
    """Tests for @intent_logger decorator"""

    def setup_method(self):
        set_current_intent(None)
        set_log_level(LogLevel.DEBUG)
        clear_intent_log()

    def test_basic_decorator(self):
        """Basic decorator wraps function"""
        @intent_logger
        def my_function():
            return 42

        result = my_function()
        assert result == 42

    def test_decorator_with_args(self):
        """Decorator works with function arguments"""
        @intent_logger
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_decorator_logs_intent(self):
        """Decorator logs intents"""
        @intent_logger
        def my_function():
            return "done"

        my_function()

        log = get_intent_log()
        assert len(log) >= 2  # started + completed

        # Find completed entry
        completed = [e for e in log if e["status"] == "completed"]
        assert len(completed) == 1
        assert completed[0]["intent_name"] == "my_function"

    def test_decorator_with_custom_name(self):
        """Decorator uses custom name"""
        @intent_logger(name="custom-name")
        def my_function():
            pass

        my_function()

        log = get_intent_log()
        assert any(e["intent_name"] == "custom-name" for e in log)

    def test_decorator_logs_args(self):
        """Decorator can log arguments"""
        @intent_logger(log_args=True)
        def greet(name):
            return f"Hello, {name}"

        greet("World")

        log = get_intent_log()
        assert any("args" in e.get("metadata", {}) for e in log)

    def test_decorator_logs_result(self):
        """Decorator can log result"""
        @intent_logger(log_result=True)
        def compute():
            return 42

        compute()

        log = get_intent_log()
        completed = [e for e in log if e["status"] == "completed"]
        assert any("result" in e.get("metadata", {}) for e in completed)

    def test_decorator_logs_exceptions(self):
        """Decorator logs exceptions"""
        @intent_logger
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        log = get_intent_log()
        failed = [e for e in log if e["status"] == "failed"]
        assert len(failed) == 1
        assert "error" in failed[0]["metadata"]

    def test_nested_decorators(self):
        """Nested decorated functions track depth (0-indexed)"""
        @intent_logger
        def outer():
            return inner()

        @intent_logger
        def inner():
            ctx = get_current_intent()
            return ctx.depth if ctx else -1

        depth = outer()
        assert depth == 1  # outer=0 (root), inner=1 (child)

    def test_decorator_respects_log_level(self):
        """Decorator respects log level"""
        set_log_level(LogLevel.IMPORTANT)
        clear_intent_log()

        @intent_logger(level=LogLevel.DEBUG)
        def debug_function():
            return "debug"

        @intent_logger(level=LogLevel.IMPORTANT)
        def important_function():
            return "important"

        debug_function()
        important_function()

        log = get_intent_log()

        # Only important should be logged
        assert not any(e["intent_name"] == "debug_function" for e in log)
        assert any(e["intent_name"] == "important_function" for e in log)


class TestAsyncIntentLogger:
    """Tests for async @intent_logger decorator"""

    def setup_method(self):
        set_current_intent(None)
        set_log_level(LogLevel.DEBUG)
        clear_intent_log()

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Async decorator works"""
        @intent_logger
        async def async_function():
            await asyncio.sleep(0.001)
            return "done"

        result = await async_function()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_async_logs_intent(self):
        """Async decorator logs intents"""
        @intent_logger
        async def async_function():
            await asyncio.sleep(0.001)
            return "done"

        await async_function()

        log = get_intent_log()
        completed = [e for e in log if e["status"] == "completed"]
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_async_exception_logging(self):
        """Async decorator logs exceptions"""
        @intent_logger
        async def failing_async():
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError):
            await failing_async()

        log = get_intent_log()
        failed = [e for e in log if e["status"] == "failed"]
        assert len(failed) == 1


class TestIntentLoggerClass:
    """Tests for @intent_logger_class decorator"""

    def setup_method(self):
        clear_intent_log()
        set_log_level(LogLevel.DEBUG)

    def test_class_decorator(self):
        """Class decorator wraps methods"""
        @intent_logger_class
        class MyService:
            def process(self):
                return "processed"

        service = MyService()
        result = service.process()

        assert result == "processed"

        log = get_intent_log()
        assert any(e["intent_name"] == "process" for e in log)

    def test_class_decorator_excludes_private(self):
        """Class decorator excludes private methods"""
        @intent_logger_class
        class MyService:
            def public_method(self):
                return "public"

            def _private_method(self):
                return "private"

        service = MyService()
        service.public_method()
        service._private_method()

        log = get_intent_log()
        assert any(e["intent_name"] == "public_method" for e in log)
        assert not any(e["intent_name"] == "_private_method" for e in log)

    def test_class_decorator_with_methods_filter(self):
        """Class decorator can filter specific methods"""
        @intent_logger_class(methods=["method_a"])
        class MyService:
            def method_a(self):
                return "a"

            def method_b(self):
                return "b"

        service = MyService()
        service.method_a()
        service.method_b()

        log = get_intent_log()
        assert any(e["intent_name"] == "method_a" for e in log)
        assert not any(e["intent_name"] == "method_b" for e in log)


class TestIntentLoggerConfig:
    """Tests for IntentLoggerConfig"""

    def test_default_config(self):
        """Default configuration values"""
        config = IntentLoggerConfig()

        assert config.name is None
        assert config.level == LogLevel.INFO
        assert config.log_args is False
        assert config.log_result is False
        assert config.log_exceptions is True
        assert config.persist is False
        assert config.sign is False

    def test_custom_config(self):
        """Custom configuration values"""
        config = IntentLoggerConfig(
            name="custom",
            level=LogLevel.CRITICAL,
            log_args=True,
            log_result=True,
            persist=True,
            sign=True,
            metadata={"key": "value"},
        )

        assert config.name == "custom"
        assert config.level == LogLevel.CRITICAL
        assert config.log_args is True
        assert config.persist is True
        assert config.metadata["key"] == "value"


class TestReasoningTemplate:
    """Tests for reasoning template feature"""

    def setup_method(self):
        clear_intent_log()
        set_log_level(LogLevel.DEBUG)

    def test_custom_reasoning_template(self):
        """Custom reasoning template is used"""
        @intent_logger(reasoning_template="Called {func} with arguments")
        def my_function():
            return 42

        my_function()

        log = get_intent_log()
        assert any("Called my_function" in e.get("reasoning", "") for e in log)


class TestPersistence:
    """Tests for persistence feature"""

    def setup_method(self):
        clear_intent_log()
        set_log_level(LogLevel.DEBUG)

    def test_persist_intent(self):
        """Intents can be persisted to storage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from intentlog.storage import IntentLogStorage

            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            @intent_logger(persist=True)
            def my_function():
                return "done"

            # Change to tmpdir so storage finds project
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                my_function()

                # Check if persisted
                chained = storage.load_chained_intents()
                assert len(chained) >= 1
            finally:
                os.chdir(old_cwd)


# ============================================================================
# Extended Context Management Tests
# ============================================================================

from intentlog.context import (
    ContextStatus,
    EnhancedIntentContextManager,
    intent_scope_enhanced,
    register_on_enter_hook,
    register_on_exit_hook,
    unregister_on_enter_hook,
    unregister_on_exit_hook,
    clear_hooks,
    propagate_context_to_env,
    restore_context_from_env,
    get_all_tags,
    get_all_labels,
    has_tag_in_chain,
    get_root_context,
    get_trace_id,
    get_span_id,
    with_tags,
    with_labels,
    INTENT_CONTEXT_ENV_VAR,
)


class TestContextStatus:
    """Tests for ContextStatus enum"""

    def test_status_values(self):
        """ContextStatus has expected values"""
        assert ContextStatus.ACTIVE.value == "active"
        assert ContextStatus.COMPLETED.value == "completed"
        assert ContextStatus.FAILED.value == "failed"
        assert ContextStatus.TIMEOUT.value == "timeout"
        assert ContextStatus.CANCELLED.value == "cancelled"


class TestIntentContextTags:
    """Tests for tags and labels on IntentContext"""

    def setup_method(self):
        set_current_intent(None)

    def test_add_single_tag(self):
        """Add single tag to context"""
        with intent_scope("tagged") as ctx:
            ctx.add_tag("api")
            assert ctx.has_tag("api")
            assert "api" in ctx.tags

    def test_add_multiple_tags(self):
        """Add multiple tags to context"""
        with intent_scope("tagged") as ctx:
            ctx.add_tags("api", "external", "critical")
            assert ctx.has_tag("api")
            assert ctx.has_tag("external")
            assert ctx.has_tag("critical")

    def test_set_label(self):
        """Set label on context"""
        with intent_scope("labeled") as ctx:
            ctx.set_label("service", "payment")
            assert ctx.get_label("service") == "payment"

    def test_get_label_default(self):
        """Get label with default value"""
        with intent_scope("labeled") as ctx:
            assert ctx.get_label("missing") is None
            assert ctx.get_label("missing", "default") == "default"

    def test_tags_in_to_dict(self):
        """Tags appear in to_dict output"""
        with intent_scope("tagged") as ctx:
            ctx.add_tags("a", "b")
            ctx.set_label("k", "v")
            d = ctx.to_dict()
            assert set(d["tags"]) == {"a", "b"}
            assert d["labels"] == {"k": "v"}


class TestContextSerialization:
    """Tests for context serialization/deserialization"""

    def setup_method(self):
        set_current_intent(None)

    def test_serialize_deserialize(self):
        """Context can be serialized and deserialized"""
        with intent_scope("original") as ctx:
            ctx.add_tags("tag1", "tag2")
            ctx.set_label("key", "value")
            encoded = ctx.serialize()

        restored = IntentContext.deserialize(encoded)
        assert restored.intent_id == ctx.intent_id
        assert restored.intent_name == "original"
        assert "tag1" in restored.tags
        assert "tag2" in restored.tags
        assert restored.labels["key"] == "value"

    def test_trace_headers(self):
        """Generate W3C trace headers"""
        with intent_scope("traced") as ctx:
            headers = ctx.to_trace_headers()
            assert "traceparent" in headers
            assert headers["x-intent-name"] == "traced"
            assert ctx.trace_id in headers["traceparent"]
            assert ctx.span_id in headers["traceparent"]

    def test_from_trace_headers(self):
        """Create context from trace headers"""
        headers = {
            "traceparent": "00-abcdef1234567890abcdef1234567890-1234567890abcdef-01",
            "x-intent-name": "received",
            "x-session-id": "session123",
        }
        ctx = IntentContext.from_trace_headers(headers)
        assert ctx.trace_id == "abcdef1234567890abcdef1234567890"
        assert ctx.intent_name == "received"
        assert ctx.session_id == "session123"


class TestContextStatusManagement:
    """Tests for context status management"""

    def setup_method(self):
        set_current_intent(None)

    def test_context_starts_active(self):
        """Context starts in active status"""
        with intent_scope("active") as ctx:
            assert ctx.status == ContextStatus.ACTIVE
            assert ctx.is_active

    def test_context_completes(self):
        """Context completes on exit"""
        with intent_scope_enhanced("completing") as ctx:
            pass
        assert ctx.status == ContextStatus.COMPLETED
        assert ctx.end_time is not None

    def test_context_fails_on_exception(self):
        """Context fails on exception"""
        try:
            with intent_scope_enhanced("failing") as ctx:
                raise ValueError("test error")
        except ValueError:
            pass
        assert ctx.status == ContextStatus.FAILED
        assert "test error" in ctx.metadata.get("error", "")

    def test_manual_complete(self):
        """Context can be manually completed"""
        ctx = IntentContext(intent_id="test", intent_name="manual")
        ctx.complete()
        assert ctx.status == ContextStatus.COMPLETED
        assert ctx.end_time is not None

    def test_manual_fail(self):
        """Context can be manually failed"""
        ctx = IntentContext(intent_id="test", intent_name="manual")
        ctx.fail("manual error")
        assert ctx.status == ContextStatus.FAILED
        assert ctx.metadata["error"] == "manual error"


class TestContextTimeout:
    """Tests for context timeout"""

    def setup_method(self):
        set_current_intent(None)

    def test_timeout_not_expired(self):
        """Context not timed out before deadline"""
        with intent_scope_enhanced("quick", timeout_seconds=60) as ctx:
            assert not ctx.is_timed_out
            assert ctx.is_active

    def test_is_timed_out_property(self):
        """is_timed_out returns True after deadline"""
        ctx = IntentContext(intent_id="test", intent_name="timeout")
        ctx.timeout_at = datetime.now() - timedelta(seconds=1)
        assert ctx.is_timed_out
        assert not ctx.is_active


class TestContextHooks:
    """Tests for context hooks system"""

    def setup_method(self):
        set_current_intent(None)
        clear_hooks()

    def teardown_method(self):
        clear_hooks()

    def test_on_enter_hook(self):
        """On enter hook is called"""
        entered = []

        def on_enter(ctx):
            entered.append(ctx.intent_name)

        register_on_enter_hook(on_enter)

        with intent_scope_enhanced("hooked"):
            pass

        assert "hooked" in entered

    def test_on_exit_hook(self):
        """On exit hook is called"""
        exited = []

        def on_exit(ctx, error):
            exited.append((ctx.intent_name, error))

        register_on_exit_hook(on_exit)

        with intent_scope_enhanced("hooked"):
            pass

        assert ("hooked", None) in exited

    def test_on_exit_hook_with_error(self):
        """On exit hook receives error"""
        exited = []

        def on_exit(ctx, error):
            exited.append((ctx.intent_name, type(error).__name__ if error else None))

        register_on_exit_hook(on_exit)

        try:
            with intent_scope_enhanced("failing"):
                raise ValueError("test")
        except ValueError:
            pass

        assert ("failing", "ValueError") in exited

    def test_unregister_hook(self):
        """Hooks can be unregistered"""
        called = []

        def hook(ctx):
            called.append(1)

        register_on_enter_hook(hook)
        assert unregister_on_enter_hook(hook)

        with intent_scope_enhanced("test"):
            pass

        assert len(called) == 0

    def test_local_hooks(self):
        """Context manager accepts local hooks"""
        entered = []
        exited = []

        with intent_scope_enhanced(
            "local",
            on_enter=lambda ctx: entered.append(ctx.intent_name),
            on_exit=lambda ctx, err: exited.append(ctx.intent_name),
        ):
            pass

        assert "local" in entered
        assert "local" in exited


class TestEnhancedContextManager:
    """Tests for EnhancedIntentContextManager"""

    def setup_method(self):
        set_current_intent(None)
        clear_hooks()

    def test_enhanced_with_tags(self):
        """Enhanced manager supports tags"""
        with intent_scope_enhanced("tagged", tags={"a", "b"}) as ctx:
            assert ctx.has_tag("a")
            assert ctx.has_tag("b")

    def test_enhanced_with_labels(self):
        """Enhanced manager supports labels"""
        with intent_scope_enhanced("labeled", labels={"k": "v"}) as ctx:
            assert ctx.get_label("k") == "v"

    def test_enhanced_with_timeout(self):
        """Enhanced manager supports timeout"""
        with intent_scope_enhanced("timed", timeout_seconds=60) as ctx:
            assert ctx.timeout_at is not None
            assert not ctx.is_timed_out


class TestEnvironmentPropagation:
    """Tests for environment variable propagation"""

    def setup_method(self):
        set_current_intent(None)
        if INTENT_CONTEXT_ENV_VAR in os.environ:
            del os.environ[INTENT_CONTEXT_ENV_VAR]

    def teardown_method(self):
        if INTENT_CONTEXT_ENV_VAR in os.environ:
            del os.environ[INTENT_CONTEXT_ENV_VAR]

    def test_propagate_to_env(self):
        """Context is propagated to environment"""
        with intent_scope("propagated") as ctx:
            propagate_context_to_env()
            assert INTENT_CONTEXT_ENV_VAR in os.environ

    def test_restore_from_env(self):
        """Context is restored from environment"""
        with intent_scope("original") as ctx:
            ctx.add_tag("test-tag")
            propagate_context_to_env()

        set_current_intent(None)
        restored = restore_context_from_env()

        assert restored is not None
        assert restored.intent_name == "original"
        assert "test-tag" in restored.tags


class TestContextQueryFunctions:
    """Tests for context query functions"""

    def setup_method(self):
        set_current_intent(None)

    def test_get_all_tags(self):
        """Get all tags from chain"""
        with intent_scope("outer") as outer:
            outer.add_tag("outer-tag")
            with intent_scope("inner") as inner:
                inner.add_tag("inner-tag")
                tags = get_all_tags()
                assert "outer-tag" in tags
                assert "inner-tag" in tags

    def test_get_all_labels(self):
        """Get all labels from chain"""
        with intent_scope("outer") as outer:
            outer.set_label("env", "prod")
            with intent_scope("inner") as inner:
                inner.set_label("service", "api")
                labels = get_all_labels()
                assert labels["env"] == "prod"
                assert labels["service"] == "api"

    def test_has_tag_in_chain(self):
        """Check tag existence in chain"""
        with intent_scope("outer") as outer:
            outer.add_tag("parent-tag")
            with intent_scope("inner"):
                assert has_tag_in_chain("parent-tag")
                assert not has_tag_in_chain("nonexistent")

    def test_get_root_context(self):
        """Get root context from chain"""
        with intent_scope("root") as root:
            with intent_scope("child"):
                with intent_scope("grandchild"):
                    found = get_root_context()
                    assert found is not None
                    assert found.intent_name == "root"

    def test_get_trace_id(self):
        """Get trace ID from current context"""
        assert get_trace_id() is None

        with intent_scope("traced") as ctx:
            assert get_trace_id() == ctx.trace_id

    def test_get_span_id(self):
        """Get span ID from current context"""
        assert get_span_id() is None

        with intent_scope("spanned") as ctx:
            assert get_span_id() == ctx.span_id


class TestContextDecorators:
    """Tests for context decorators"""

    def setup_method(self):
        set_current_intent(None)

    def test_with_tags_decorator(self):
        """@with_tags adds tags to context"""
        @with_tags("decorated", "tagged")
        def my_func():
            ctx = get_current_intent()
            return ctx.tags if ctx else set()

        with intent_scope("outer") as ctx:
            tags = my_func()
            assert "decorated" in ctx.tags
            assert "tagged" in ctx.tags

    def test_with_labels_decorator(self):
        """@with_labels adds labels to context"""
        @with_labels(service="test", version="1.0")
        def my_func():
            ctx = get_current_intent()
            return ctx.labels if ctx else {}

        with intent_scope("outer") as ctx:
            my_func()
            assert ctx.get_label("service") == "test"
            assert ctx.get_label("version") == "1.0"


class TestDistributedTracing:
    """Tests for distributed tracing support"""

    def setup_method(self):
        set_current_intent(None)

    def test_trace_id_propagates(self):
        """Trace ID propagates to children"""
        with intent_scope("parent") as parent:
            with intent_scope("child") as child:
                assert child.trace_id == parent.trace_id
                assert child.span_id != parent.span_id

    def test_parent_span_id(self):
        """Parent span ID is accessible"""
        with intent_scope("parent") as parent:
            with intent_scope("child") as child:
                assert child.parent_span_id == parent.span_id
