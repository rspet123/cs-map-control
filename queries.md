## Queries 

- Area Control -> Winrate
```
WITH area_control AS (
    SELECT
        ars.demo_id,
        dm.map_name,
        ars.round_num,
        ars.area_id,
        rs.team_a_side,
        -- Remap A/B to T/CT based on which side team A was on
        CASE WHEN rs.team_a_side = 'T'
            THEN ars.ticks_a_ctrl / NULLIF(ars.ticks_sampled, 0)
            ELSE ars.ticks_b_ctrl / NULLIF(ars.ticks_sampled, 0)
        END AS t_ctrl_ratio,
        CASE WHEN rs.team_a_side = 'CT'
            THEN ars.ticks_a_ctrl / NULLIF(ars.ticks_sampled, 0)
            ELSE ars.ticks_b_ctrl / NULLIF(ars.ticks_sampled, 0)
        END AS ct_ctrl_ratio,
        -- Remap winner to T/CT as well
        CASE WHEN r.winner_team = 'A' AND rs.team_a_side = 'T' THEN 1.0
             WHEN r.winner_team = 'B' AND rs.team_a_side = 'CT' THEN 1.0
             ELSE 0.0
        END AS t_won,
        CASE WHEN r.winner_team = 'A' AND rs.team_a_side = 'CT' THEN 1.0
             WHEN r.winner_team = 'B' AND rs.team_a_side = 'T' THEN 1.0
             ELSE 0.0
        END AS ct_won
    FROM mc_area_round_stats ars
    JOIN mc_rounds r
        ON ars.demo_id = r.demo_id
        AND ars.round_num = r.round_num
    JOIN demo_matches dm
        ON ars.demo_id = dm.demo_id
    JOIN mc_round_sides rs
        ON ars.demo_id = rs.demo_id
        AND ars.round_num = rs.round_num
    WHERE r.winner_team IS NOT NULL
),
area_win_corr AS (
    SELECT
        map_name,
        area_id,
        COUNT(*) AS rounds_present,
        (AVG(t_ctrl_ratio * t_won) - AVG(t_ctrl_ratio) * AVG(t_won))
            / NULLIF(STD(t_ctrl_ratio) * STD(t_won), 0) AS t_win_corr,
        (AVG(ct_ctrl_ratio * ct_won) - AVG(ct_ctrl_ratio) * AVG(ct_won))
            / NULLIF(STD(ct_ctrl_ratio) * STD(ct_won), 0) AS ct_win_corr,
        AVG(t_ctrl_ratio)  AS avg_t_ctrl,
        AVG(ct_ctrl_ratio) AS avg_ct_ctrl
    FROM area_control
    GROUP BY map_name, area_id
)
SELECT
    map_name,
    area_id,
    rounds_present,
    ROUND(avg_t_ctrl, 3)                           AS avg_t_ctrl,
    ROUND(avg_ct_ctrl, 3)                          AS avg_ct_ctrl,
    ROUND(t_win_corr, 3)                           AS t_win_corr,
    ROUND(ct_win_corr, 3)                          AS ct_win_corr,
    ROUND((t_win_corr + ct_win_corr) / 2, 3)       AS importance_score
FROM area_win_corr
WHERE rounds_present >= 20
ORDER BY map_name, importance_score DESC;
```

- Player Contribution - MC Score, By Side

```
WITH player_stats AS (
    SELECT
        p.player_id,
        p.steamid,
        p.name,
        -- Derive side: if player's team matches team_a_side, use that, else flip
        CASE
            WHEN mdt.team = 'A' THEN rs.team_a_side
            ELSE CASE WHEN rs.team_a_side = 'T' THEN 'CT' ELSE 'T' END
        END AS side,
        COUNT(DISTINCT CONCAT(pr.demo_id, '-', pr.round_num)) AS rounds_played,
        AVG(pr.avg_active_pct)          AS avg_active_pct,
        AVG(pr.avg_unique_pct)          AS avg_unique_pct,
        AVG(pr.avg_denial_pct)          AS avg_denial_pct,
        AVG(pr.passive_attributed_pct)  AS avg_passive_pct,
        AVG(pr.death_impact_pct)        AS avg_death_impact_pct,
        AVG(pr.round_alive_pct)         AS avg_alive_pct,
        AVG(CASE WHEN pr.survived THEN 1.0 ELSE 0.0 END) AS survival_rate
    FROM mc_player_rounds pr
    JOIN players p
        ON pr.player_id = p.player_id
    JOIN mc_demo_teams mdt
        ON pr.demo_id = mdt.demo_id
        AND pr.player_id = mdt.player_id
    JOIN mc_round_sides rs
        ON pr.demo_id = rs.demo_id
        AND pr.round_num = rs.round_num
    GROUP BY p.player_id, p.steamid, p.name, side
    HAVING COUNT(DISTINCT CONCAT(pr.demo_id, '-', pr.round_num)) >= 20
),
stats AS (
    SELECT
        side,
        AVG(avg_active_pct)         AS mean_active,  STD(avg_active_pct)         AS std_active,
        AVG(avg_passive_pct)        AS mean_passive, STD(avg_passive_pct)        AS std_passive,
        AVG(avg_denial_pct)         AS mean_denial,  STD(avg_denial_pct)         AS std_denial,
        AVG(avg_death_impact_pct)   AS mean_death,   STD(avg_death_impact_pct)   AS std_death
    FROM player_stats
    GROUP BY side
)
SELECT
    ps.steamid,
    ps.name,
    ps.side,
    ps.rounds_played,
    ROUND(ps.avg_active_pct, 3)         AS avg_active_pct,
    ROUND(ps.avg_passive_pct, 3)        AS avg_passive_pct,
    ROUND(ps.avg_unique_pct, 3)         AS avg_unique_pct,
    ROUND(ps.avg_denial_pct, 3)         AS avg_denial_pct,
    ROUND(ps.avg_death_impact_pct, 3)   AS avg_death_impact_pct,
    ROUND(ps.avg_alive_pct, 3)          AS avg_alive_pct, -- Average percent of ROUND TIME alive
    ROUND(ps.survival_rate, 3)          AS survival_rate,
    ROUND(ps.avg_active_pct / NULLIF(ps.avg_active_pct + ps.avg_passive_pct, 0), 3) AS aggression_ratio,
    ROUND(
          (ps.avg_active_pct       - s.mean_active)  / NULLIF(s.std_active,  0)
        + (ps.avg_passive_pct      - s.mean_passive) / NULLIF(s.std_passive, 0)
        + (ps.avg_denial_pct       - s.mean_denial)  / NULLIF(s.std_denial,  0)
        - (ps.avg_death_impact_pct - s.mean_death)   / NULLIF(s.std_death,   0)
    , 3) AS mc_score
FROM player_stats ps
JOIN stats s
    ON ps.side = s.side
ORDER BY ps.side, mc_score DESC;
```

- Win-Loss MC Diff per Player

```
SELECT
    p.name,
    ROUND(AVG(CASE WHEN r.winner_team = pr.team 
        THEN (pr.avg_unique_pct + pr.passive_attributed_pct + pr.avg_denial_pct) END), 3) AS contrib_in_wins,
    ROUND(AVG(CASE WHEN r.winner_team != pr.team 
        THEN (pr.avg_unique_pct + pr.passive_attributed_pct + pr.avg_denial_pct) END), 3) AS contrib_in_losses,
    ROUND(AVG(CASE WHEN r.winner_team = pr.team 
        THEN (pr.avg_unique_pct + pr.passive_attributed_pct + pr.avg_denial_pct) END) -
          AVG(CASE WHEN r.winner_team != pr.team 
        THEN (pr.avg_unique_pct + pr.passive_attributed_pct + pr.avg_denial_pct) END), 3) AS win_loss_diff
FROM mc_player_rounds pr
JOIN players p ON pr.player_id = p.player_id
JOIN mc_rounds r ON pr.demo_id = r.demo_id AND pr.round_num = r.round_num
GROUP BY p.player_id, p.name, p.steamid
HAVING COUNT(*) >= 10
ORDER BY win_loss_diff DESC;
```