-- =============================================================================
-- metric_summary.sql  —  Per-metric statistics queries
-- =============================================================================
-- Sections are delimited by:  -- [section_name]
-- Each section is executed independently by commands.py.
-- Parameterised queries accept :experiment_name ('' = no filter)
-- and :key (required for key_detail).

-- [all_keys]
SELECT   m.key,
         COUNT(DISTINCT m.run_id)          AS runs,
         COUNT(*)                          AS total_steps,
         ROUND(MIN(m.value), 4)            AS min_val,
         ROUND(MAX(m.value), 4)            AS max_val,
         ROUND(AVG(m.value), 4)            AS mean_val,
         MAX(m.step) - MIN(m.step)         AS step_span
FROM     metrics     m
JOIN     runs        r ON r.id = m.run_id
JOIN     experiments e ON e.id = r.experiment_id
WHERE    (:experiment_name = '' OR e.name = :experiment_name)
GROUP BY m.key
ORDER BY runs DESC, total_steps DESC;

-- [key_detail]
SELECT   m.run_id,
         e.name                            AS experiment,
         ROUND(MIN(m.value), 4)            AS min_val,
         ROUND(MAX(m.value), 4)            AS max_val,
         ROUND(AVG(m.value), 4)            AS mean_val,
         COUNT(*)                          AS steps
FROM     metrics     m
JOIN     runs        r ON r.id = m.run_id
JOIN     experiments e ON e.id = r.experiment_id
WHERE    m.key = :key
  AND    (:experiment_name = '' OR e.name = :experiment_name)
GROUP BY m.run_id
ORDER BY mean_val ASC;
