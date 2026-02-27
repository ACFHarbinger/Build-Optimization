-- =============================================================================
-- prune_candidates.sql  —  Select runs eligible for pruning
-- =============================================================================
-- Named parameters (all required):
--   :cutoff            ISO-8601 timestamp; only runs *before* this are returned
--   :status            run status to match, or the literal string 'all'
--   :experiment_name   experiment name to restrict to, or '' for any

SELECT r.id,
       r.status,
       r.run_type,
       r.start_time,
       e.name AS experiment_name
FROM   runs        r
JOIN   experiments e ON r.experiment_id = e.id
WHERE  r.start_time        <  :cutoff
  AND  (:status            = 'all' OR r.status = :status)
  AND  (:experiment_name   = ''    OR e.name   = :experiment_name)
ORDER  BY r.start_time ASC;
