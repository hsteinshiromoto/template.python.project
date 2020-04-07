SELECT DISTINCT {date_column}
    ,{target_column}
    ,COUNT({predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Count"
    ,SUM({predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) As "Sum"
    ,AVG({predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Average"
    ,STDDEV({predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Sum"
    ,MIN({predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Min"
    ,MAX({predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Max"
    ,PERCENTILE_CONT(0.75) WITH GROUP (ORDER BY {predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Q3"
    ,PERCENTILE_CONT(0.5) WITH GROUP (ORDER BY {predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Median"
    ,PERCENTILE_CONT(0.25) WITH GROUP (ORDER BY {predictor_column}) OVER (PARTITION BY {date_column}, {target_column}) AS "Q1"
FROM {schema}.{table}