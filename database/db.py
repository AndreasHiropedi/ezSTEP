import json
import os
import sqlite3
import threading
import time

from config import DATABASE_URL


# Extract database path from URL
def get_database_path():
    if DATABASE_URL.startswith("sqlite:///"):
        path = DATABASE_URL[10:]  # Remove 'sqlite:///' prefix
    elif DATABASE_URL.startswith("sqlite://"):
        path = DATABASE_URL[9:]  # Remove 'sqlite://' prefix
    else:
        path = DATABASE_URL or "ezstep.db"  # Use as-is or default

    # Ensure directory exists for absolute paths
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    return path


# Global variables
_db_path = get_database_path()
_local = threading.local()


def _get_connection():
    """Get database connection with automatic transaction management"""
    if not hasattr(_local, "connection"):
        _local.connection = sqlite3.connect(
            _db_path, check_same_thread=False, timeout=30.0
        )
        _local.connection.row_factory = sqlite3.Row
    return _local.connection


def _init_database():
    """Initialize database with required tables"""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                data TEXT,
                last_access_time INTEGER,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """
        )

        # Create index for cleanup performance
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_last_access 
            ON user_sessions(last_access_time)
        """
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise


def store_user_session_data(session_id, data):
    """Store user session data in SQLite."""
    data_json = json.dumps(data)
    current_time = int(time.time())

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_sessions 
            (session_id, data, last_access_time) 
            VALUES (?, ?, ?)
        """,
            (session_id, data_json, current_time),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_user_session_data(session_id):
    """Retrieve user session data from SQLite."""
    current_time = int(time.time())

    conn = _get_connection()
    try:
        # Update last access time
        conn.execute(
            """
            UPDATE user_sessions 
            SET last_access_time = ? 
            WHERE session_id = ?
        """,
            (current_time, session_id),
        )

        # Get the data
        cursor = conn.execute(
            """
            SELECT data FROM user_sessions 
            WHERE session_id = ?
        """,
            (session_id,),
        )

        row = cursor.fetchone()
        conn.commit()

        if row and row["data"]:
            return json.loads(row["data"])
        else:
            raise Exception(f"No data found for session {session_id}")
    except Exception:
        conn.rollback()
        raise


def cleanup_old_sessions(time_limit_seconds=1800):
    """Clean up old session data."""
    current_time = int(time.time())
    cutoff_time = current_time - time_limit_seconds

    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            DELETE FROM user_sessions 
            WHERE last_access_time < ?
        """,
            (cutoff_time,),
        )

        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count
    except Exception:
        conn.rollback()
        raise


# Initialize database on import
_init_database()
