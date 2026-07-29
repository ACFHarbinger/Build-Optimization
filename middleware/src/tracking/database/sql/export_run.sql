-- =============================================================================
-- export_run.sql  —  Full run data export queries for export_run.py
-- =============================================================================
-- All sections share the named parameter  :run_id.
-- Python executes each section in order and assembles the result into JSON.

-- [run_detail]
SELECT r.id,
       r.name,
       r.status,
       r.run_type,
       r.start_time,
       r.end_time,
       r.artifact_dir,
       r.error_message,
       e.name AS experiment_name
FROM   runs        r
JOIN   experiments e ON r.experiment_id = e.id
WHERE  r.id = :run_id;

-- [tags]
SELECT key,
       value
FROM   run_tags
WHERE  run_id = :run_id
ORDER  BY key;

-- [params]
SELECT key,
       value        -- JSON-encoded; decoded by Python
FROM   params
WHERE  run_id = :run_id
ORDER  BY key;

-- [metrics]
-- Grouped in Python by key → list of {step, value, timestamp}.
SELECT key,
       value,
       step,
       timestamp
FROM   metrics
WHERE  run_id = :run_id
ORDER  BY key,
          step;

-- [artifacts]
SELECT id,
       name,
       path,
       artifact_type,
       file_hash,
       size_bytes,
       created_at,
       metadata     -- JSON-encoded
FROM   artifacts
WHERE  run_id = :run_id
ORDER  BY created_at;

-- [dataset_events]
SELECT id,
       event_type,
       file_path,
       file_hash,
       prev_hash,
       size_bytes,
       num_samples,
       metadata,    -- JSON-encoded
       timestamp
FROM   dataset_events
WHERE  run_id = :run_id
ORDER  BY timestamp;
