"""
Cross-platform file locking for IntentLog.

Provides file locking that works on both Unix (fcntl) and Windows (msvcrt).
Falls back to a simple lock file approach if neither is available.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# Platform-specific imports
LOCK_AVAILABLE = False
LOCK_BACKEND = "none"

if sys.platform == "win32":
    try:
        import msvcrt
        LOCK_AVAILABLE = True
        LOCK_BACKEND = "msvcrt"
    except ImportError:
        pass
else:
    try:
        import fcntl
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


class FileLock:
    """
    Cross-platform file lock implementation.

    Usage:
        with FileLock(path, exclusive=True):
            # Do work with file
            pass

    On Unix: Uses fcntl.flock() for advisory locking
    On Windows: Uses msvcrt.locking() for mandatory locking
    Fallback: Uses a .lock file for basic locking
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
        self._file = None
        self._lock_file = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def acquire(self) -> None:
        """Acquire the file lock."""
        if LOCK_BACKEND == "fcntl":
            self._acquire_fcntl()
        elif LOCK_BACKEND == "msvcrt":
            self._acquire_msvcrt()
        else:
            self._acquire_fallback()

    def release(self) -> None:
        """Release the file lock."""
        if LOCK_BACKEND == "fcntl":
            self._release_fcntl()
        elif LOCK_BACKEND == "msvcrt":
            self._release_msvcrt()
        else:
            self._release_fallback()

    def _acquire_fcntl(self) -> None:
        """Acquire lock using fcntl (Unix)."""
        import fcntl

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file (create if doesn't exist)
        self._file = open(self.path, "a+")

        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH

        start_time = time.time()
        while True:
            try:
                fcntl.flock(self._file.fileno(), lock_type | fcntl.LOCK_NB)
                return
            except (IOError, OSError):
                if time.time() - start_time >= self.timeout:
                    self._file.close()
                    self._file = None
                    raise FileLockTimeout(
                        f"Could not acquire lock on {self.path} within {self.timeout}s"
                    )
                time.sleep(self.poll_interval)

    def _release_fcntl(self) -> None:
        """Release lock using fcntl (Unix)."""
        import fcntl

        if self._file:
            try:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            finally:
                self._file.close()
                self._file = None

    def _acquire_msvcrt(self) -> None:
        """Acquire lock using msvcrt (Windows)."""
        import msvcrt

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file (create if doesn't exist)
        self._file = open(self.path, "a+")

        # msvcrt.LK_NBLCK = non-blocking exclusive lock
        # msvcrt.LK_NBRLCK = non-blocking shared lock (not available, use exclusive)
        lock_mode = msvcrt.LK_NBLCK

        start_time = time.time()
        while True:
            try:
                # Lock the first byte of the file
                msvcrt.locking(self._file.fileno(), lock_mode, 1)
                return
            except (IOError, OSError):
                if time.time() - start_time >= self.timeout:
                    self._file.close()
                    self._file = None
                    raise FileLockTimeout(
                        f"Could not acquire lock on {self.path} within {self.timeout}s"
                    )
                time.sleep(self.poll_interval)

    def _release_msvcrt(self) -> None:
        """Release lock using msvcrt (Windows)."""
        import msvcrt

        if self._file:
            try:
                # Unlock the first byte
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self._file.close()
                self._file = None

    def _acquire_fallback(self) -> None:
        """Acquire lock using a lock file (fallback)."""
        lock_path = Path(str(self.path) + self.LOCK_SUFFIX)

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        while True:
            try:
                # Try to create lock file exclusively
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._lock_file = lock_path
                return
            except FileExistsError:
                # Check if the process holding the lock is still alive
                if self._is_stale_lock(lock_path):
                    try:
                        os.unlink(lock_path)
                        continue
                    except OSError:
                        pass

                if time.time() - start_time >= self.timeout:
                    raise FileLockTimeout(
                        f"Could not acquire lock on {self.path} within {self.timeout}s"
                    )
                time.sleep(self.poll_interval)

    def _release_fallback(self) -> None:
        """Release lock by removing lock file (fallback)."""
        if self._lock_file and self._lock_file.exists():
            try:
                os.unlink(self._lock_file)
            except OSError:
                pass
            self._lock_file = None

    def _is_stale_lock(self, lock_path: Path) -> bool:
        """Check if a lock file is stale (process no longer exists)."""
        try:
            with open(lock_path, "r") as f:
                pid = int(f.read().strip())
            # Check if process exists
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError, PermissionError, OSError):
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
