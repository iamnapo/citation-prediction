import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx.GraphLoader
import org.apache.spark.graphx.lib.PageRank
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.{typedLit, udf}
import org.apache.spark.sql.{SaveMode, SparkSession}
import utils.Utils

object FeatureExtraction {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Feature Extraction").master("local[*]").getOrCreate()
    val sc = ss.sparkContext
    import ss.implicits._

    // Read initial files
    val pre_training_set_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = " ")
      .option("inferSchema", "true")
      .csv("src/main/resources/training_set.txt")
    val pre_testing_set_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = " ")
      .option("inferSchema", "true")
      .csv("src/main/resources/testing_set.txt")
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

    var post_training_set_df = pre_training_set_df
      .withColumn("cosine_similarity", cosineSimilarity($"Target", $"Source"))
      .withColumn("num_of_same_words_in_title", countSameWordsInTitle($"Target", $"Source"))
      .withColumn("num_of_same_words_in_abstract", countSameWordsInAbstract($"Target", $"Source"))
      .withColumn("have_same_authors", haveSameAuthors($"Target", $"Source"))
      .withColumn("year_gap", yearGap($"Target", $"Source"))
    var post_testing_set_df = pre_testing_set_df
      .withColumn("cosine_similarity", cosineSimilarity($"Target", $"Source"))
      .withColumn("num_of_same_words_in_title", countSameWordsInTitle($"Target", $"Source"))
      .withColumn("num_of_same_words_in_abstract", countSameWordsInAbstract($"Target", $"Source"))
      .withColumn("have_same_authors", haveSameAuthors($"Target", $"Source"))
      .withColumn("year_gap", yearGap($"Target", $"Source"))
      .withColumn("label", getLabel($"Target", $"Source"))

    //pagerank
    val graph = GraphLoader.edgeListFile(sc, "src/main/resources/training_set_edges.csv").cache()
    val pageRank = PageRank.runUntilConvergence(graph, 0.0001)
      .vertices.map(p => (p._1, p._2)).collect()

    def sumPageRanks = udf((id_1: String, id_2: String) => {
      val pageRank_1 = pageRank.filter(_._1.toString == id_1).map(_._2)
      val pageRank_2 = pageRank.filter(_._1.toString == id_2).map(_._2)
      var rank_1 = 0.0
      if (pageRank_1.length > 0) rank_1 = pageRank_1.take(1)(0)
      var rank_2 = 0.0
      if (pageRank_2.length > 0) rank_2 = pageRank_2.take(1)(0)
      rank_1 + rank_2
    })

    post_training_set_df = post_training_set_df
      .withColumn("pagerank", sumPageRanks($"Target", $"Source"))
    post_testing_set_df = post_testing_set_df
      .withColumn("pagerank", sumPageRanks($"Target", $"Source"))


    // degrees
    val indegrees = graph.inDegrees.map(p => (p._1, p._2)).collect()
    val outDegrees = graph.outDegrees.map(p => (p._1, p._2)).collect()
    val degrees = graph.degrees.map(p => (p._1, p._2)).collect()

    def nodeDegree = udf((id_1: String, in_or_out: String) => {
      if (in_or_out == "in") {
        val deg = indegrees.filter(_._1.toString == id_1).map(_._2)
        if (deg.length > 0) {
          deg.take(1)(0)
        } else {
          0
        }
      } else {
        val deg = outDegrees.filter(_._1.toString == id_1).map(_._2)
        if (deg.length > 0) {
          deg.take(1)(0)
        } else {
          0
        }
      }
    })

    def neighbourSimilarity = udf((id_1: String, id_2: String) => {
      try {
        val deg_1 = degrees.filter(_._1.toString == id_1).map(_._2).take(1)(0)
        val deg_2 = degrees.filter(_._1.toString == id_2).map(_._2).take(1)(0)
        deg_1 * deg_2
      } catch {
        case _: Exception => 0
      }
    })

    post_training_set_df = post_training_set_df
      .withColumn("source_indegree", nodeDegree($"Source", typedLit("in")))
      .withColumn("target_indegree", nodeDegree($"Target", typedLit("in")))
      .withColumn("source_outdegree", nodeDegree($"Source", typedLit("out")))
      .withColumn("target_outdegree", nodeDegree($"Target", typedLit("out")))
      .withColumn("neighbour_similarity", neighbourSimilarity($"Target", $"Source"))
    post_testing_set_df = post_testing_set_df
      .withColumn("source_indegree", nodeDegree($"Source", typedLit("in")))
      .withColumn("target_indegree", nodeDegree($"Target", typedLit("in")))
      .withColumn("source_outdegree", nodeDegree($"Source", typedLit("out")))
      .withColumn("target_outdegree", nodeDegree($"Target", typedLit("out")))
      .withColumn("neighbour_similarity", neighbourSimilarity($"Target", $"Source"))

    post_testing_set_df
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .option("codec", "gzip")
      .csv("src/main/resources/dataset/training_set_with_features.csv")

    post_training_set_df
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .option("codec", "gzip")
      .csv("src/main/resources/dataset/testing_set_with_features.csv")
  }
}
