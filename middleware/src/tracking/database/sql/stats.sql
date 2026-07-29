-- =============================================================================
-- stats.sql  —  Comprehensive database statistics queries
-- =============================================================================
-- Sections are delimited by:  -- [section_name]
-- Each section is executed independently by commands.py.
-- Parameterised queries accept :experiment_name ('' = no filter).

-- [table_sizes]
SELECT 'experiments'     AS table_name, COUNT(*) AS rows FROM experiments
UNION ALL
SELECT 'runs',           COUNT(*) FROM runs
UNION ALL
SELECT 'run_tags',       COUNT(*) FROM run_tags
UNION ALL
SELECT 'params',         COUNT(*) FROM params
UNION ALL
SELECT 'metrics',        COUNT(*) FROM metrics
UNION ALL
SELECT 'artifacts',      COUNT(*) FROM artifacts
UNION ALL
SELECT 'dataset_events', COUNT(*) FROM dataset_events;

-- [experiment_stats]
SELECT e.name                                                                     AS experiment,
       COUNT(r.id)                                                                AS total_runs,
       SUM(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END)                   AS completed,
       SUM(CASE WHEN r.status = 'failed'    THEN 1 ELSE 0 END)                   AS failed,
       SUM(CASE WHEN r.status = 'running'   THEN 1 ELSE 0 END)                   AS running,
       ROUND(
           AVG(
               CASE
                   WHEN r.end_time IS NOT NULL
                   THEN (julianday(r.end_time) - julianday(r.start_time)) * 86400.0
               END
           ), 1
       )                                                                          AS avg_duration_s
FROM   experiments e
LEFT   JOIN runs r ON r.experiment_id = e.id
WHERE  (:experiment_name = '' OR e.name = :experiment_name)
GROUP  BY e.id, e.name
ORDER  BY total_runs DESC;

-- [top_metrics]
SELECT   m.key,
         COUNT(DISTINCT m.run_id)   AS runs_tracking,
         COUNT(*)                   AS total_steps,
         ROUND(MIN(m.value), 4)     AS min_val,
         ROUND(MAX(m.value), 4)     AS max_val,
         ROUND(AVG(m.value), 4)     AS mean_val
FROM     metrics     m
JOIN     runs        r ON r.id = m.run_id
JOIN     experiments e ON e.id = r.experiment_id
WHERE    (:experiment_name = '' OR e.name = :experiment_name)
GROUP BY m.key
ORDER BY runs_tracking DESC, total_steps DESC
LIMIT    20;

-- [artifact_type_stats]
SELECT   a.artifact_type,
         COUNT(*)                        AS count,
         COALESCE(SUM(a.size_bytes), 0)  AS total_bytes
FROM     artifacts   a
JOIN     runs        r ON r.id = a.run_id
JOIN     experiments e ON e.id = r.experiment_id
WHERE    (:experiment_name = '' OR e.name = :experiment_name)
GROUP BY a.artifact_type
ORDER BY count DESC;

-- [dataset_event_stats]
SELECT   de.event_type,
         COUNT(*) AS count
FROM     dataset_events de
JOIN     runs           r  ON r.id  = de.run_id
JOIN     experiments    e  ON e.id  = r.experiment_id
WHERE    (:experiment_name = '' OR e.name = :experiment_name)
GROUP BY de.event_type
ORDER BY count DESC;

-- [run_duration_stats]
SELECT COUNT(*)                                                                     AS finished_runs,
       ROUND(MIN((julianday(r.end_time) - julianday(r.start_time)) * 86400.0), 1)  AS min_s,
       ROUND(MAX((julianday(r.end_time) - julianday(r.start_time)) * 86400.0), 1)  AS max_s,
       ROUND(AVG((julianday(r.end_time) - julianday(r.start_time)) * 86400.0), 1)  AS mean_s
FROM   runs        r
JOIN   experiments e ON e.id = r.experiment_id
WHERE  r.end_time IS NOT NULL
  AND  (:experiment_name = '' OR e.name = :experiment_name);

-- [run_activity]
SELECT DATE(r.start_time) AS day,
       COUNT(*)            AS runs
FROM   runs        r
JOIN   experiments e ON e.id = r.experiment_id
WHERE  r.start_time >= DATE('now', '-30 days')
  AND  (:experiment_name = '' OR e.name = :experiment_name)
GROUP BY day
ORDER BY day DESC;
