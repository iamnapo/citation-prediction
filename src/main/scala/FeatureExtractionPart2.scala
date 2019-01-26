import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{SaveMode, SparkSession}
import utils.Utils

object FeatureExtractionPart2 {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Feature Extraction Part 2").master("local[*]").getOrCreate()
    import ss.implicits._

    // Read initial files
    val pairs_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = " ")
      .option("inferSchema", "true")
      .csv("src/main/resources/pairs.txt")
    val pre_node_information_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = ",")
      .option("inferSchema", "true")
      .csv("src/main/resources/node_information.csv")
      .na.fill(Map("Authors" -> "Unknown Authors", "Title" -> "Unknown Title", "Abstract" -> "Unknown Abstract"))
    val ground_truth_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = " ")
      .option("inferSchema", "true")
      .csv("src/main/resources/Cit-HepTh.txt")
    val ground_truth_rdd = ground_truth_df.rdd.collect()

    var post_node_information_df = pre_node_information_df

    println("Starting feature extraction")
    val t0 = System.nanoTime()

    //tokenize titles
    val tokenizer = new Tokenizer()
      .setInputCol("Title")
      .setOutputCol("Title_tmp_1")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("Title_tmp_1")
      .setOutputCol("Title_tmp_2")
      .setCaseSensitive(false)
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))

    post_node_information_df = stopWordsRemover
      .transform(tokenizer
        .transform(pre_node_information_df))
    post_node_information_df = post_node_information_df.drop("Title").drop("Title_tmp_1")
    post_node_information_df = post_node_information_df.withColumnRenamed("Title_tmp_2", "Title")

    //tokenize abstracts
    tokenizer.setInputCol("Abstract").setOutputCol("Abstract_tmp_1")
    stopWordsRemover.setInputCol("Abstract_tmp_1").setOutputCol("Abstract_tmp_2")

    post_node_information_df = stopWordsRemover
      .setInputCol("Abstract_tmp_1")
      .setOutputCol("Abstract_tmp_2")
      .transform(tokenizer
        .setInputCol("Abstract")
        .setOutputCol("Abstract_tmp_1")
        .transform(post_node_information_df))
    post_node_information_df = post_node_information_df.drop("Abstract").drop("Abstract_tmp_1")
    post_node_information_df = post_node_information_df.withColumnRenamed("Abstract_tmp_2", "Abstract")

    val counter = new CountVectorizer().setInputCol("Abstract").setOutputCol("Abstract_Counts").setBinary(true)
    post_node_information_df = counter.fit(post_node_information_df).transform(post_node_information_df)

    val post_node_information_rdd = post_node_information_df.rdd.collect()

    def cosineSimilarity = udf((id_1: String, id_2: String) => {
      val vector_1 = post_node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[org.apache.spark.ml.linalg.SparseVector]("Abstract_Counts"))
        .take(1)(0)
      val vector_2 = post_node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[org.apache.spark.ml.linalg.SparseVector]("Abstract_Counts"))
        .take(1)(0)
      new Utils().cosineSimilarity(vector_1.toArray.map(_.toInt), vector_2.toArray.map(_.toInt))
    })

    def yearGap = udf((id_1: String, id_2: String) => {
      val year_1 = post_node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[Int]("Year"))
        .take(1)(0)
      val year_2 = post_node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[Int]("Year"))
        .take(1)(0)
      year_1 - year_2
    })

    def getLabel = udf((id_1: String, id_2: String) => {
      val count = ground_truth_rdd
        .count(row => {
          row.getInt(0) == id_1.toInt && row.getInt(1) == id_2.toInt
        })
      if (count > 0) {
        1
      } else {
        0
      }
    })

    def countSameWordsInTitle = udf((id_1: String, id_2: String) => {
      val title_1 = post_node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[Seq[String]]("Title"))
        .take(1)(0)
      val title_2 = post_node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[Seq[String]]("Title"))
        .take(1)(0)
      var count = 0
      title_1.foreach(word => {
        if (title_2.contains(word)) {
          count += 1
        }
      })
      count
    })

    def countSameWordsInAbstract = udf((id_1: String, id_2: String) => {
      val title_1 = post_node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[Seq[String]]("Abstract"))
        .take(1)(0)
      val title_2 = post_node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[Seq[String]]("Abstract"))
        .take(1)(0)
      var count = 0
      title_1.foreach(word => {
        if (title_2.contains(word)) {
          count += 1
        }
      })
      count
    })

    def haveSameAuthors = udf((id_1: String, id_2: String) => {
      val authors_1 = post_node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[String]("Authors"))
        .take(1)(0)
      val authors_2 = post_node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[String]("Authors"))
        .take(1)(0)
      var contains = 0
      if ((authors_1 contains authors_2) || (authors_2 contains authors_1)) {
        contains = 1
      }
      contains
    })

    val pairs_df_with_features = pairs_df
      .withColumn("cosine_similarity", cosineSimilarity($"Target", $"Source"))
      .withColumn("num_of_same_words_in_title", countSameWordsInTitle($"Target", $"Source"))
      .withColumn("num_of_same_words_in_abstract", countSameWordsInAbstract($"Target", $"Source"))
      .withColumn("have_same_authors", haveSameAuthors($"Target", $"Source"))
      .withColumn("year_gap", yearGap($"Target", $"Source"))
      .withColumn("label", getLabel($"Target", $"Source"))
      .cache()

    pairs_df_with_features
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .option("codec", "gzip")
      .csv("src/main/resources/dataset/pairs_with_features.csv")

    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1E9).toInt + " seconds")
  }
}
