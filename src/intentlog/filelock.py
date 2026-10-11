"""
Cross-platform file locking for IntentLog.

Provides file locking that works on both Unix (fcntl) and Windows (msvcrt).
Falls back to a simple lock file approach if neither is available.

Locks are re-entrant per thread: a thread that already holds a lock on a
path can acquire it again (for example a nested shared read inside an
exclusive read-modify-write) without blocking on itself. The OS-level lock
is only released when the outermost holder releases it. Different threads
still exclude each other, because each thread opens its own file handle.
"""

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Dict, Optional

# Platform-specific imports
LOCK_AVAILABLE = False
LOCK_BACKEND = "none"

if sys.platform == "win32":
    try:
        import msvcrt  # noqa: F401
        LOCK_AVAILABLE = True
        LOCK_BACKEND = "msvcrt"
    except ImportError:
        pass
else:
    try:
        import fcntl  # noqa: F401
        LOCK_AVAILABLE = True
        LOCK_BACKEND = "fcntl"
    except ImportError:
        pass


class FileLockError(Exception):
    """Base exception for file locking errors."""
    pass


class FileLockTimeout(FileLockError):
    """Raised when lock acquisition times out."""
    pass


class _LockRecord:
    """
    Bookkeeping for one OS-level lock physically held by a thread.

    Several FileLock instances in the same thread may share one record
    (re-entrant acquisition); ``count`` tracks the outstanding holders and
    the OS resources are released when it drops to zero.
    """

    __slots__ = ("key", "exclusive", "count", "registry", "file", "lock_file")

    def __init__(self, key: str, exclusive: bool, registry: Dict[str, "_LockRecord"]):
        self.key = key
        self.exclusive = exclusive
        self.count = 0
        self.registry = registry
        self.file: Optional[IO[str]] = None  # Open file object (fcntl / msvcrt backends)
        self.lock_file: Optional[Path] = None  # Created lock file (fallback backend)


_thread_state = threading.local()


def _held_locks() -> Dict[str, _LockRecord]:
    """Return the registry of locks held by the current thread (keyed by path)."""
    registry = getattr(_thread_state, "held", None)
    if registry is None:
        registry = {}
        _thread_state.held = registry
    return registry


def _lock_key(path: Path) -> str:
    """Normalize a path so that nested requests for the same file match."""
    return os.path.normcase(os.path.abspath(str(path)))


class FileLock:
    """
    Cross-platform, per-thread re-entrant file lock implementation.

    Usage:
        with FileLock(path, exclusive=True):
            # Do work with file
            pass

    On Unix: Uses fcntl.flock() for advisory locking
    On Windows: Uses msvcrt.locking() for mandatory locking
    Fallback: Uses a .lock file for basic locking

    Re-entrancy: if the current thread already holds a lock on ``path``, a
    nested acquisition (exclusive or shared inside exclusive, shared inside
    shared) succeeds immediately and the nested release does not drop the
    outer lock. Requesting an exclusive lock while the thread only holds a
    shared one raises FileLockError rather than deadlocking against itself.
    Other threads and processes are still excluded by the OS-level lock.
    """

    # Lock file suffix for fallback mode
    LOCK_SUFFIX = ".lock"

    def __init__(
        self,
        path: Path,
        exclusive: bool = True,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ):
        """
        Initialize file lock.

        Args:
            path: Path to the file to lock
            exclusive: True for exclusive (write) lock, False for shared (read) lock
            timeout: Maximum time to wait for lock (seconds)
            poll_interval: Time between lock attempts (seconds)
        """
        self.path = Path(path)
        self.exclusive = exclusive
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._record: Optional[_LockRecord] = None
        self._depth = 0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    @property
    def is_locked(self) -> bool:
        """True while this instance holds (or shares) the lock."""
        return self._depth > 0

    def acquire(self) -> None:
        """Acquire the file lock (re-entrant within the current thread)."""
        registry = _held_locks()
        key = _lock_key(self.path)

        record = registry.get(key)
        if record is not None:
            if self.exclusive and not record.exclusive:
                raise FileLockError(
                    f"Cannot upgrade shared lock on {self.path} to exclusive "
                    "while it is held by this thread"
                )
            record.count += 1
            self._record = record
            self._depth += 1
            return

        record = _LockRecord(key, self.exclusive, registry)
        if LOCK_BACKEND == "fcntl":
            self._acquire_fcntl(record)
        elif LOCK_BACKEND == "msvcrt":
            self._acquire_msvcrt(record)
        else:
            self._acquire_fallback(record)

        record.count = 1
        registry[key] = record
        self._record = record
        self._depth += 1

    def release(self) -> None:
        """Release the file lock (the OS lock is dropped by the last holder)."""
        record = self._record
        if record is None:
            return

        self._depth -= 1
        if self._depth <= 0:
            self._depth = 0
            self._record = None

        record.count -= 1
        if record.count > 0:
            return

        record.registry.pop(record.key, None)
        if LOCK_BACKEND == "fcntl":
            self._release_fcntl(record)
        elif LOCK_BACKEND == "msvcrt":
            self._release_msvcrt(record)
        else:
            self._release_fallback(record)

    def _open_handle(self, record: _LockRecord) -> IO[str]:
        """Open (creating if needed) the file whose handle carries the lock."""
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            handle = open(self.path, "a+")
        except OSError as e:
            raise FileLockError(f"Failed to open file for locking: {self.path}: {e}") from e

        record.file = handle
        return handle

    def _acquire_fcntl(self, record: _LockRecord) -> None:
        """Acquire lock using fcntl (Unix)."""
        import fcntl

        handle = self._open_handle(record)

        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH

        start_time = time.monotonic()
        while True:
            try:
                fcntl.flock(handle.fileno(), lock_type | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() - start_time >= self.timeout:
                    handle.close()
                    record.file = None
                    raise FileLockTimeout(
                        f"Could not acquire lock on {self.path} within {self.timeout}s"
                    )
                time.sleep(self.poll_interval)

    def _release_fcntl(self, record: _LockRecord) -> None:
        """Release lock using fcntl (Unix)."""
        import fcntl

        handle = record.file
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                record.file = None

    def _acquire_msvcrt(self, record: _LockRecord) -> None:
        """Acquire lock using msvcrt (Windows)."""
        import msvcrt

        handle = self._open_handle(record)

        # msvcrt.LK_NBLCK = non-blocking exclusive lock
        # msvcrt.LK_NBRLCK = non-blocking shared lock (not available, use exclusive)
        lock_mode = msvcrt.LK_NBLCK

        start_time = time.monotonic()
        while True:
            try:
                # Lock the first byte of the file
                handle.seek(0)
                msvcrt.locking(handle.fileno(), lock_mode, 1)
                return
            except OSError:
                if time.monotonic() - start_time >= self.timeout:
                    handle.close()
                    record.file = None
                    raise FileLockTimeout(
                        f"Could not acquire lock on {self.path} within {self.timeout}s"
                    )
                time.sleep(self.poll_interval)

    def _release_msvcrt(self, record: _LockRecord) -> None:
        """Release lock using msvcrt (Windows)."""
        import msvcrt

        handle = record.file
        if handle is not None:
            try:
                # Unlock the first byte
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                handle.close()
                record.file = None

    def _acquire_fallback(self, record: _LockRecord) -> None:
        """Acquire lock using a lock file (fallback)."""
        lock_path = Path(str(self.path) + self.LOCK_SUFFIX)

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.monotonic()
        while True:
            try:
                # Try to create lock file exclusively
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                record.lock_file = lock_path
                return
            except FileExistsError:
                # Check if the process holding the lock is still alive
                if self._is_stale_lock(lock_path):
                    try:
                        os.unlink(lock_path)
                        continue
                    except OSError:
                        pass

                if time.monotonic() - start_time >= self.timeout:
                    raise FileLockTimeout(
                        f"Could not acquire lock on {self.path} within {self.timeout}s"
                    )
                time.sleep(self.poll_interval)

    def _release_fallback(self, record: _LockRecord) -> None:
        """Release lock by removing lock file (fallback)."""
        if record.lock_file and record.lock_file.exists():
            try:
                os.unlink(record.lock_file)
            except OSError:
                pass
        record.lock_file = None

    def _is_stale_lock(self, lock_path: Path) -> bool:
        """Check if a lock file is stale (process no longer exists)."""
        try:
            with open(lock_path) as f:
                content = f.read().strip()
        except OSError:
            # Vanished (owner released it) or unreadable: let the caller retry.
            return False

        if not content:
            # The owner creates the file before writing its pid; an empty
            # file is a live lock whose owner has not finished writing yet.
            return False

        try:
            pid = int(content)
        except ValueError:
            return True

        try:
            # Check if process exists
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            return True
        except PermissionError:
            # Process exists but belongs to another user: still live.
            return False
        except OSError:
            return True


@contextmanager
def file_lock(
    path: Path,
    exclusive: bool = True,
    timeout: float = 10.0,
):
    """
    Context manager for file locking.

    Args:
        path: Path to the file to lock
        exclusive: True for exclusive lock, False for shared lock
        timeout: Maximum time to wait for lock

    Usage:
        with file_lock(Path("file.json"), exclusive=True):
            # Do work with file
            pass
    """
    lock = FileLock(path, exclusive=exclusive, timeout=timeout)
    try:
        lock.acquire()
        yield lock
    finally:
        lock.release()


def lock_file_for_write(path: Path, timeout: float = 10.0) -> FileLock:
    """
    Convenience function for exclusive file locking.

    Args:
        path: Path to lock
        timeout: Lock timeout in seconds

    Returns:
        FileLock context manager
    """
    return FileLock(path, exclusive=True, timeout=timeout)


def lock_file_for_read(path: Path, timeout: float = 10.0) -> FileLock:
    """
    Convenience function for shared file locking.

    Args:
        path: Path to lock
        timeout: Lock timeout in seconds

    Returns:
        FileLock context manager
    """
    return FileLock(path, exclusive=False, timeout=timeout)
