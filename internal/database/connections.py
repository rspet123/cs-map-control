import os

import logfire
import pymysql.cursors
from dotenv import load_dotenv

load_dotenv()
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


class PooledCursor:
    """
    Wraps a PyMySQL cursor and the pool-proxied connection it came from.

    * Forwards every attribute access to the underlying cursor (execute,
      fetchone, fetchall, lastrowid, rowcount, description, …).
    * close() releases the underlying connection back to the SQLAlchemy pool.
    * __del__ calls close() as a safety-net for callers that skip it (several
      repositories omit an explicit close()).
    """

    def __init__(self, cursor, raw_conn):
        self._cursor = cursor
        self._raw_conn = raw_conn
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._cursor.close()
            finally:
                self._raw_conn.close()  # returns the connection to the pool

    def __del__(self):
        self.close()


class DatabasePool:
    """
    Thin facade around a SQLAlchemy Engine that exposes the same
    cursor-based interface repositories already use:

        cursor  = self.db_connection.cursor()
        cursor  = self.db_connection.cursor(dictionary=True)
        self.db_connection.commit()
        self.db_connection.rollback()

    commit() and rollback() are intentional no-ops — autocommit is enabled
    at the PyMySQL level via connect_args, so every statement commits
    immediately (identical behaviour to the old mysql-connector setup).
    """

    def __init__(self, engine):
        self._engine = engine

    # ------------------------------------------------------------------
    # Public interface (matches what repositories expect)
    # ------------------------------------------------------------------

    def cursor(self, dictionary=False):
        """Return a PooledCursor backed by a connection from the pool."""
        raw_conn = self._engine.raw_connection()
        cur = raw_conn.cursor(
            pymysql.cursors.DictCursor if dictionary else pymysql.cursors.Cursor
        )
        return PooledCursor(cur, raw_conn)

    def commit(self):
        pass  # autocommit=True — no explicit commit needed

    def rollback(self):
        pass  # autocommit=True — single-statement ops cannot be rolled back

    def __bool__(self):
        return True


class DBConnections:
    """
    Manages the SQLAlchemy QueuePool for prefire_db.

    Replaces the old hand-rolled ConnectionProxy / ping-reconnect logic.
    SQLAlchemy's pool_pre_ping=True handles idle-timeout reconnection
    automatically before each connection is handed out.
    """

    def __init__(self):
        self.prefire_db = None

    def connect_prefire_db(self):
        """
        Build the SQLAlchemy engine and wrap it in a DatabasePool.

        Returns:
            bool: True on success, False on failure.
        """
        try:
            url = URL.create(
                drivername="mysql+pymysql",
                username=os.getenv("DB_USER", "root"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "3306")),
                database=os.getenv("DB_NAME", "statistics"),
            )
            print(f"Connecting to database at {url}...")
            engine = create_engine(
                url,
                # One connection always ready, up to 10 extra under load.
                pool_size=5,
                max_overflow=10,
                # Emit a lightweight SELECT 1 before handing out a connection
                # that has been idle — replaces the old manual ping dance.
                pool_pre_ping=True,
                # Pass autocommit directly to PyMySQL so every statement
                # commits immediately (mirrors old autocommit=True behaviour).
                connect_args={"autocommit": True},
            )
            self.prefire_db = DatabasePool(engine)
            return True
        except Exception as e:
            logfire.error("Error connecting to database", error=str(e))
            return False

    def get_prefire_db(self):
        """Return the DatabasePool (for callers that use get_prefire_db())."""
        return self.prefire_db

    def is_connected(self):
        """
        True once the engine has been initialised.

        pool_pre_ping=True means the pool self-heals; we don't need to probe
        the connection here — the before_request hook just needs to know
        whether connect_prefire_db() has been called successfully.
        """
        return self.prefire_db is not None

    def close(self):
        """Dispose the connection pool, closing all idle connections."""
        if self.prefire_db is not None:
            self.prefire_db._engine.dispose()
            self.prefire_db = None
            logfire.info("PrefireDB connection pool closed")

    def connect_all(self):
        """Establish all database connections (extendable for future DBs)."""
        return self.connect_prefire_db()

# ------------------------------------------------------------------
if __name__ == "__main__":
    db_connections = DBConnections()
    if db_connections.connect_all():
        print("Successfully connected to prefire_db.")
    else:
        print("Failed to connect to prefire_db.")