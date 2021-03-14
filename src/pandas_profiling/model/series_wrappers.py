import attr

from pandas_profiling.config import config as config


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

    def __init__(self, series, sample, persist=True):
        from pyspark.sql.functions import array, map_keys, map_values
        from pyspark.sql.types import MapType

        self.series = series
        # if series type is dict, handle that separately
        if isinstance(series.schema[0].dataType, MapType):
            self.series = series.select(
                array(map_keys(series[self.name]), map_values(series[self.name])).alias(
                    self.name
                )
            )

        self.persist_bool = persist
        self.series.persist()

        # TODO this needs to be computed before dropna
        self.n_rows = self.series.count()

        # if dropna is not handled at the dataframe level, we must handle at series level
        if config["spark"]["dropna"].get(str) == "series":
            self.series = self.series.na.drop()
            self.series.persist()

        # compute useful statistics once
        self.dropna_count = self.series.count()
        if config["vars"]["common"]["distinct"].get(bool):
            self.distinct = self.series.distinct().count()
        if config["vars"]["common"]["unique"].get(bool):
            self.unique = self.series.dropDuplicates().count()

        self.sample = sample

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
        value_counts = self.series.groupBy(self.name).count()
        value_counts.persist()
        return value_counts

    def count_na(self):
        return self.n_rows - self.dropna_count

    def __len__(self):
        return self.n_rows

    def persist(self):
        if self.persist_bool:
            self.series.persist()

    def unpersist(self):
        if self.persist_bool:
            self.series.unpersist()
