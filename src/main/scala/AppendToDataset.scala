import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{SaveMode, SparkSession}

object AppendToDataset {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Feature Extraction").master("local[*]").getOrCreate()
    import ss.implicits._

    val t0 = System.nanoTime()

    val testing_set_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = ",")
      .option("inferSchema", "true")
      .csv("src/main/resources/testing_set_with_features.csv")

    val node_information_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = ",")
      .option("inferSchema", "true")
      .csv("src/main/resources/node_information.csv")
      .na.fill(Map("Authors" -> "Unknown Authors", "Title" -> "Unknown Title", "Abstract" -> "Unknown Abstract", "Year" -> 0))
    val node_information_rdd = node_information_df.rdd.collect()


    def haveSameAuthors = udf((id_1: String, id_2: String) => {
      val authors_1 = node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[String]("Authors"))
        .take(1)(0)
      val authors_2 = node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[String]("Authors"))
        .take(1)(0)
      var contains = 0
      if ((authors_1 contains authors_2) || (authors_2 contains authors_1)) {
        contains = 1
      }
      contains
    })

    def yearGap = udf((id_1: String, id_2: String) => {
      val year_1 = node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[Int]("Year"))
        .take(1)(0)
      val year_2 = node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[Int]("Year"))
        .take(1)(0)
      year_1 - year_2
    })

    testing_set_df
      .withColumn("year_gap", yearGap($"Target", $"Source"))
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .option("codec", "gzip")
      .csv("src/main/resources/dataset/training_set_with_features_new.csv")

    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1E9).toInt + " seconds")


  }
}
