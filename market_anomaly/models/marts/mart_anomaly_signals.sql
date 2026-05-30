-- mart_anomaly_signals.sql
-- Final analytical table combining features with prediction history.
-- This is the table Grafana queries and the API reads from.
-- Materialized as a TABLE so queries are fast.

with features as (
    select * from {{ ref('int_price_features') }}
),

predictions as (
    select
        snapshot_id::text,
        is_anomaly,
        confidence::numeric      as confidence,
        z_score::numeric         as model_z_score,
        iso_score::numeric       as iso_score,
        model_version,
        predicted_at
    from {{ source('raw', 'predictions') }}
),

joined as (
    select
        f.snapshot_id,
        f.ticker,
        f.asset_class,
        f.price,
        f.price_delta,
        f.price_delta_pct,
        f.rolling_20_mean,
        f.rolling_20_std,
        f.rolling_5_mean,
        f.z_score                               as feature_z_score,
        f.short_term_trend,
        f.day_range_pct,
        f.captured_at,

        -- prediction data (may be null if not yet scored)
        p.is_anomaly,
        p.confidence,
        p.model_z_score,
        p.iso_score,
        p.model_version,
        p.predicted_at,

        -- convenience flag
        coalesce(p.is_anomaly, false)           as is_flagged

    from features f
    left join predictions p
        on f.snapshot_id = p.snapshot_id
)

select * from joined
order by captured_at desc