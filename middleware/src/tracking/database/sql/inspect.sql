-- =============================================================================
-- inspect.sql  —  Read-only overview queries for inspect_db.py
-- =============================================================================
-- Sections are delimited by:  -- [section_name]
-- Each section is executed independently by inspect_db.py.

-- [experiments]
SELECT name,
       created_at
FROM   experiments
ORDER  BY created_at DESC;

-- [runs_by_status]
SELECT status,
       COUNT(*) AS count
FROM   runs
GROUP  BY status
ORDER  BY status;

-- [runs_by_type]
SELECT run_type,
       COUNT(*) AS count
FROM   runs
GROUP  BY run_type
ORDER  BY run_type;

-- [record_counts]
SELECT (SELECT COUNT(*) FROM metrics)        AS metric_rows,
       (SELECT COUNT(*) FROM params)         AS param_rows,
       (SELECT COUNT(*) FROM artifacts)      AS artifact_rows,
       (SELECT COUNT(*) FROM dataset_events) AS dataset_event_rows;

-- [recent_runs]
SELECT r.id,
       r.status,
       r.run_type,
       r.start_time,
       e.name AS experiment_name
FROM   runs        r
JOIN   experiments e ON r.experiment_id = e.id
ORDER  BY r.start_time DESC
LIMIT  10;
