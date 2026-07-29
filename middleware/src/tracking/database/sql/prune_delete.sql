-- =============================================================================
-- prune_delete.sql  —  Delete all data for one run (FK-safe order)
-- =============================================================================
-- Each section deletes from one table using the positional parameter  ?  bound
-- to the run_id.  Python iterates sections in file order to satisfy FK deps.

-- [delete_metrics]
DELETE FROM metrics
WHERE  run_id = ?;

-- [delete_dataset_events]
DELETE FROM dataset_events
WHERE  run_id = ?;

-- [delete_artifacts]
DELETE FROM artifacts
WHERE  run_id = ?;

-- [delete_params]
DELETE FROM params
WHERE  run_id = ?;

-- [delete_run_tags]
DELETE FROM run_tags
WHERE  run_id = ?;

-- [delete_run]
-- Must come last; child rows must already be gone.
DELETE FROM runs
WHERE  id = ?;
