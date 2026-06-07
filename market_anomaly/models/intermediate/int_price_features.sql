-- int_price_features.sql
-- Computes rolling statistical features per ticker.
-- This is the feature engineering layer the model reads from.

with snapshots as (
    select * from {{ ref('stg_price_snapshots') }}
),

with_row_number as (
    select
        *,
        row_number() over (
            partition by ticker
            order by captured_at
        ) as row_num
    from snapshots
),

with_lag as (
    select
        snapshot_id,
        ticker,
        asset_class,
        price,
        volume,
        captured_at,
        pct_from_prev_close,
        day_range_pct,

        -- price delta from previous snapshot
        lag(price) over (
            partition by ticker order by captured_at
        )                                               as prev_snapshot_price,

        price - lag(price) over (
            partition by ticker order by captured_at
        )                                               as price_delta,

        round(
            (
                (price - lag(price) over (
                    partition by ticker order by captured_at
                ))
                / nullif(lag(price) over (
                    partition by ticker order by captured_at
                ), 0) * 100
            )::numeric, 4
        )                                               as price_delta_pct,

        -- rolling 20-period stats
        avg(price) over (
            partition by ticker
            order by captured_at
            rows between 19 preceding and current row
        )                                               as rolling_20_mean,

        stddev(price) over (
            partition by ticker
            order by captured_at
            rows between 19 preceding and current row
        )                                               as rolling_20_std,

        -- rolling 5-period mean for short-term trend
        avg(price) over (
            partition by ticker
            order by captured_at
            rows between 4 preceding and current row
        )                                               as rolling_5_mean,

        -- count of captures in last 20 rows
        count(*) over (
            partition by ticker
            order by captured_at
            rows between 19 preceding and current row
        )                                               as rolling_count,

        -- baseline for the z-score excludes the current row (no look-in leakage)
        avg(price) over (
            partition by ticker
            order by captured_at
            rows between 20 preceding and 1 preceding
        )                                               as baseline_mean,

        stddev(price) over (
            partition by ticker
            order by captured_at
            rows between 20 preceding and 1 preceding
        )                                               as baseline_std,

        -- volume baseline (excludes current row), mirrors the price baseline
        avg(volume) over (
            partition by ticker
            order by captured_at
            rows between 20 preceding and 1 preceding
        )                                               as baseline_vol_mean,

        stddev(volume) over (
            partition by ticker
            order by captured_at
            rows between 20 preceding and 1 preceding
        )                                               as baseline_vol_std,

        row_num
    from with_row_number
),

with_zscore as (
    select
        *,
        case
            when baseline_std > 0
            then round(
                ((price - baseline_mean) / baseline_std)::numeric, 4
            )
            else 0
        end                                             as z_score,

        -- volume z-score, mirrors the Python volume_z feature
        case
            when baseline_vol_std > 0
            then round(
                ((volume - baseline_vol_mean) / baseline_vol_std)::numeric, 4
            )
            else 0
        end                                             as volume_z,

        -- trend direction: price above or below short-term mean
        case
            when price > rolling_5_mean then 'above'
            when price < rolling_5_mean then 'below'
            else 'at'
        end                                             as short_term_trend

    from with_lag
)

select * from with_zscore
where row_num >= 3  -- need at least 3 rows for meaningful features