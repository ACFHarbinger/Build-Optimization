-- =============================================================================
-- clean.sql  —  Delete all tracking data while preserving the schema
-- =============================================================================
-- Executed as a single script (no bind parameters).
-- Deletion order respects FK constraints (children before parents).

DELETE FROM metrics;
DELETE FROM dataset_events;
DELETE FROM artifacts;
DELETE FROM params;
DELETE FROM run_tags;
DELETE FROM runs;
DELETE FROM experiments;

-- Reclaim the freed pages so the file shrinks immediately.
VACUUM;
