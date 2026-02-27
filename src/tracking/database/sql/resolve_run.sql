-- =============================================================================
-- resolve_run.sql  —  Resolve a run ID from a partial UUID or "latest" lookup
-- =============================================================================
-- Used by export_run.py to turn a prefix or an experiment name into a full UUID.
-- Named parameter:  :prefix            (for [by_prefix])
--                   :experiment_name   (for [latest], '' = any experiment)

-- [by_prefix]
-- Returns the most-recent run whose UUID starts with :prefix.
SELECT id
FROM   runs
WHERE  id LIKE :prefix
ORDER  BY start_time DESC
LIMIT  1;

-- [latest]
-- Returns the most-recent run, optionally filtered to one experiment.
SELECT r.id
FROM   runs        r
JOIN   experiments e ON r.experiment_id = e.id
WHERE  (:experiment_name = '' OR e.name = :experiment_name)
ORDER  BY r.start_time DESC
LIMIT  1;
