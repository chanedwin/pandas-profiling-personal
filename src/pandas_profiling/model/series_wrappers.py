import attr


@attr.s
class Sample:
    id = attr.ib()
    data = attr.ib()
    name = attr.ib()
    caption = attr.ib(default=None)


class SparkSeries:
    """
    A lot of optimisations left to do (persisting, caching etc), but when functionality completed
    """

    def __init__(self, series, persist=True):
        from pyspark.sql.functions import array, map_keys, map_values
        from pyspark.sql.types import MapType

        self.series = series
        self.dropna = self.series.na.drop()
        # if series type is dict, handle that separately
        if isinstance(series.schema[0].dataType, MapType):
            self.series = series.select(
                array(map_keys(series[self.name]), map_values(series[self.name])).alias(
                    self.name
                )
            )

        self.persist_bool = persist
        if self.persist_bool:
            series.persist()
            self.dropna.persist()

        # compute useful statistics once
        self.n_rows = self.series.count()
        self.dropna_count = self.dropna.count()
        self.distinct = self.dropna.distinct().count()
        self.unique = self.dropna.dropDuplicates().count()

    @property
    def type(self):
        return self.series.schema.fields[0].dataType

    @property
    def name(self):
        return self.series.columns[0]

    @property
    def empty(self) -> bool:
        return self.n_rows == 0

    def value_counts(self):
        """

        Args:
            n: by default, get only 1000

        Returns:

        """

        from pyspark.sql.functions import array, map_keys, map_values
        from pyspark.sql.types import MapType

        # if series type is dict, handle that separately
        if isinstance(self.series.schema[0].dataType, MapType):
            new_df = self.dropna.groupby(
                map_keys(self.series[self.name]).alias("key"),
                map_values(self.series[self.name]).alias("value"),
            ).count()
            value_counts = (
                new_df.withColumn(self.name, array(new_df["key"], new_df["value"]))
                .select(self.name, "count")
                .orderBy("count", ascending=False)
            )
        else:
            value_counts = self.dropna.groupBy(self.name).count()
        value_counts.persist()
        return value_counts

    def count_na(self):
        return self.n_rows - self.dropna_count

    def __len__(self):
        return self.n_rows

    def get_spark_series(self):
        return self.series

    def persist(self):
        if self.persist_bool:
            self.series.persist()

    def unpersist(self):
        if self.persist_bool:
            self.series.unpersist()
