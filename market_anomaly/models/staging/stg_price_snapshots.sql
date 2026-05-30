-- stg_price_snapshots.sql
-- Clean and normalize raw price snapshots.
-- Casts types, renames for consistency, filters out bad rows.

with source as (
    select * from {{ source('raw', 'price_snapshots') }}
),

cleaned as (
    select
        id::text                                    as snapshot_id,
        ticker,
        asset_type,
        price::numeric                              as price,
        volume::numeric                             as volume,
        prev_close::numeric                         as prev_close,
        day_high::numeric                           as day_high,
        day_low::numeric                            as day_low,
        captured_at::timestamptz                    as captured_at,

        -- derived
        case
            when ticker like '%-USD' then 'crypto'
            else 'equity'
        end                                         as asset_class,

        -- price vs previous close
        case
            when prev_close > 0
            then round(((price - prev_close) / prev_close * 100)::numeric, 4)
            else null
        end                                         as pct_from_prev_close,

        -- day range width as % of price
        case
            when price > 0
            then round(((day_high - day_low) / price * 100)::numeric, 4)
            else null
        end                                         as day_range_pct

    from source
    where
        price is not null
        and price > 0
        and captured_at is not null
)

select * from cleaned