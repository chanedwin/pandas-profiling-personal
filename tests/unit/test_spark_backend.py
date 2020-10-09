import pytest

from pandas_profiling.utils.common import test_for_pyspark_pyarrow_incompatibility, VERSION_WARNING

@pytest.mark.sparktest
def test_import_spark_session(spark_session):
    """
    Test if its possible to import spark
    """
    try:
        import pyspark
        from pyspark import SparkConf, SparkContext
        from pyspark.sql import SparkSession
    except ImportError:
        pytest.fail(
            """Could not import pyspark - is SPARK_HOME and JAVA_HOME set as variables?
                    see https://spark.apache.org/docs/latest/quick-start.html and ensure
                    that your spark instance is configured properly"""
        )


@pytest.mark.sparktest
def test_create_spark_session(spark_session):
    """
    Test if pytest-spark's spark sessions can be properly created
    """
    try:
        from pyspark.sql import SparkSession

        assert isinstance(spark_session, SparkSession)
    except AssertionError:
        pytest.fail(
            """pytest spark_session was not configured properly and could not be created
        is pytest-spark installed and configured properly?"""
        )

@pytest.mark.sparktest
def test_spark_config_check(spark_session,monkeypatch):
    """
    test_for_pyspark_pyarrow_incompatibility
    """
    import pyspark
    import pyarrow
    import os
    monkeypatch.setattr(pyspark, "__version__", "2.3.0")
    monkeypatch.setattr(pyspark, "__version__", "0.15.0")
    monkeypatch.setattr(os, "environ", {})
    with pytest.warns(VERSION_WARNING):
        test_for_pyspark_pyarrow_incompatibility()

    monkeypatch.setattr(pyspark, "__version__", "2.3.0")
    monkeypatch.setattr(pyspark, "__version__", "0.15.0")
    monkeypatch.setattr(os, "environ", {"ARROW_PRE_0_15_IPC_FORMAT":0})
    with pytest.warns(VERSION_WARNING):
        test_for_pyspark_pyarrow_incompatibility()

    spar
    if spark_version[0] == "2" and spark_version[1] in ["3", "4"]:

        pyarrow_version = pyarrow.__version__.split(".")

        # this checks if the env has an incompatible arrow version (not < 0.15)
        if not (pyarrow_version[0] == "0" and int(pyarrow_version[1]) < 15):
            # if os variable is not set, likely unhandled
            if "ARROW_PRE_0_15_IPC_FORMAT" not in os.environ:
                warnings.warn(VERSION_WARNING)
