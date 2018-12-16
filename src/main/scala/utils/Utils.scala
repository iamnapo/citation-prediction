package utils

import org.apache.spark.sql.SparkSession

  class Utils {

    def createEdgeListFile(): Unit = {
      val ss = SparkSession.builder().appName("Link Prediction").master("local[4]").getOrCreate()
      import ss.implicits._

      val pre_training_set_df = ss.read
        .option("header", value = true)
        .option("delimiter", value = " ")
        .csv("src/main/resources/training_set.txt")

      val training_set_edges = pre_training_set_df.rdd
        .filter(row => row.getString(2) == "1")
        .map(row => (row.getString(0), row.getString(1)))
        .toDF

      training_set_edges
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .save("src/main/resources/training_set_edges.csv")
    }

  }