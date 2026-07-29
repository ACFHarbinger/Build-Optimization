-- =============================================================================
-- compact.sql  —  Integrity check and storage compaction
-- =============================================================================
-- Three sections executed sequentially by compact_db.py.
-- integrity_check must pass before the other two are run.

-- [integrity_check]
-- Returns a single row: 'ok' on success, or a description of the corruption.
PRAGMA integrity_check;

-- [wal_checkpoint]
-- Merge the WAL file back into the main database file and truncate it.
-- This must run before VACUUM to avoid a size *increase* post-VACUUM.
PRAGMA wal_checkpoint(TRUNCATE);

-- [vacuum]
-- Rewrite the database file, reclaiming all freed pages.
VACUUM;
