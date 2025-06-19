import duckdb
from typing import List,Union
import datetime as dt
from pathlib import Path


def generate_main_database(
    loc_ids: List[int],
    start_date: dt.datetime,
    end_date: dt.datetime,
    output_dir: Union[str,Path],
    database: str = ':memory:'
) -> None:

    # Format loc_ids for SQL VALUES
    loc_values = ', '.join(f"({loc})" for loc in loc_ids)

    sql = f"""
    WITH locs AS (
      SELECT * FROM (VALUES {loc_values}) AS t(loc_id)
    ),
    params AS (
      SELECT
        '{start_date}'::timestamp AS p_start,
        '{end_date}'::timestamp   AS p_end
    ),
    hours AS (
      SELECT
        l.loc_id,
        series AS date
      FROM locs l
      CROSS JOIN generate_series(
        (SELECT p_start FROM params),
        (SELECT p_end   FROM params),
        INTERVAL '1 hour'
      ) AS series
    ),
    feadate AS (
      SELECT
        loc_id,
        EXTRACT(year  FROM date) AS year,
        EXTRACT(month FROM date) AS month,
        EXTRACT(day   FROM date) AS day,
        EXTRACT(hour  FROM date) AS hour,

        sin(2 * pi() * EXTRACT(doy  FROM date) / 365.25) AS sin_doy,
        cos(2 * pi() * EXTRACT(doy  FROM date) / 365.25) AS cos_doy,
        sin(2 * pi() * EXTRACT(hour FROM date) / 24)      AS sin_hod,
        cos(2 * pi() * EXTRACT(hour FROM date) / 24)      AS cos_hod,

        EXTRACT(dow FROM date) AS day_of_week
      FROM hours
    )
    SELECT * FROM feadate;
    """

    con = duckdb.connect(database=database)
    con.execute("INSTALL parquet;")
    con.execute("LOAD parquet;")

    con.execute(f"CREATE OR REPLACE VIEW feadate AS {sql}")
    con.execute(
        f"COPY feadate TO '{output_dir}' \
         (FORMAT PARQUET, PARTITION_BY (loc_id, year, month, day, hour))"
    )
    con.close()

