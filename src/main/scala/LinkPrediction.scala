import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx.GraphLoader
import org.apache.spark.graphx.lib.PageRank
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{typedLit, udf}


object LinkPrediction {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val NUM_TO_TRAIN = 10000
    val NUM_TO_TEST = 1000

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Link Prediction").master("local[*]").getOrCreate()
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
    val ground_truth_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = " ")
      .option("inferSchema", "true")
      .csv("src/main/resources/Cit-HepTh.txt")

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
        .transform(pre_node_information_df.na.fill(Map("Title" -> "Unknown Title"))))
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
        .transform(post_node_information_df.na.fill(Map("Abstract" -> "Unknown Abstract"))))
    post_node_information_df = post_node_information_df.drop("Abstract").drop("Abstract_tmp_1")
    post_node_information_df = post_node_information_df.withColumnRenamed("Abstract_tmp_2", "Abstract")

    val hashingTF = new HashingTF().setInputCol("Abstract").setOutputCol("Abstract_Hash").setNumFeatures(64)
    val idf = new IDF().setInputCol("Abstract_Hash").setOutputCol("Abstract_Hash_IDF")
    val normalizer = new Normalizer().setInputCol("Abstract_Hash_IDF").setOutputCol("Abstract_Vector")

    post_node_information_df = hashingTF.transform(post_node_information_df)
    post_node_information_df = idf.fit(post_node_information_df).transform(post_node_information_df)
    post_node_information_df = normalizer.transform(post_node_information_df)
    post_node_information_df = post_node_information_df.drop("Abstract_Hash").drop("Abstract_Hash_IDF")

    val post_node_information_rdd = post_node_information_df.rdd.collect()

    def dotProduct = udf((id_1: String, id_2: String) => {
      val vector_1 = post_node_information_rdd.filter(row => row.getInt(0) == id_1.toInt)
        .map(row => row.getAs[org.apache.spark.ml.linalg.SparseVector]("Abstract_Vector"))
        .take(1)(0)
      val vector_2 = post_node_information_rdd.filter(row => row.getInt(0) == id_2.toInt)
        .map(row => row.getAs[org.apache.spark.ml.linalg.SparseVector]("Abstract_Vector"))
        .take(1)(0)
      var b = 0.0
      val these = vector_1.toDense.values.iterator
      val those = vector_2.toDense.values.iterator
      while (these.hasNext && those.hasNext)
        b += these.next() * those.next()
      b
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

    var post_training_set_df = pre_training_set_df
      .withColumn("cosine_similarity", dotProduct($"Target", $"Source"))
      .withColumn("num_of_same_words_in_title", countSameWordsInTitle($"Target", $"Source"))
      .withColumn("num_of_same_words_in_abstract", countSameWordsInAbstract($"Target", $"Source"))
    var post_testing_set_df = pre_testing_set_df
      .withColumn("cosine_similarity", dotProduct($"Target", $"Source"))
      .withColumn("num_of_same_words_in_title", countSameWordsInTitle($"Target", $"Source"))
      .withColumn("num_of_same_words_in_abstract", countSameWordsInAbstract($"Target", $"Source"))

    //pagerank
    val graph = GraphLoader.edgeListFile(sc, "src/main/resources/training_set_edges.csv")
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

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "Source",
        "Target",
        "num_of_same_words_in_title",
        "num_of_same_words_in_abstract",
        "source_indegree",
        "target_indegree",
        "source_outdegree",
        "target_outdegree",
        "cosine_similarity",
        "neighbour_similarity"
      ))
      .setOutputCol("features")

    val trainingData = assembler
      .transform(post_training_set_df.withColumnRenamed("Edge", "label").limit(NUM_TO_TRAIN))
      .select("label", "features")
      .cache()

    val ground_truth_rdd = ground_truth_df.rdd.collect()


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

    val testData = assembler
      .transform(post_testing_set_df.withColumn("label", getLabel($"Target", $"Source")).limit(NUM_TO_TEST))
      .select("label", "features")

    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1e9).toInt + " seconds")
    println("Starting model training")
    val t2 = System.nanoTime()
    // Train SVM model.
    //    val model = new LinearSVC().setMaxIter(1000).setRegParam(0.1).fit(trainingData)
    //    model.write.overwrite().save("src/main/models/LSVC")

    // Train Logistic Regression model
    val model = new LogisticRegression().setMaxIter(1000).setTol(1E-7).setFitIntercept(true).fit(trainingData)
    model.write.overwrite().save("src/main/models/LogisticRegression")

    // Train Neural Network
    //    val model = new MultilayerPerceptronClassifier().setLayers(Array[Int](10, 5, 4, 2)).setSeed(1234L).setMaxIter(1000).fit(trainingData)
    //    model.write.overwrite().save("src/main/models/NeuralNetwork")

    val t3 = System.nanoTime()
    println("Elapsed time: " + ((t3 - t2) / 1E9 / 60).toInt + " minutes")
    println("Starting model Prediction")
    val t4 = System.nanoTime()


    // Predict
    val predictions = model.transform(testData)
    val scoreAndLabels = predictions.select("prediction", "label")
      .rdd
      .map(p => (p.getDouble(0), p.getInt(1).toDouble))
      .cache()

    val metrics = new MulticlassMetrics(scoreAndLabels)

    println(metrics.confusionMatrix)
    println(s"Accuracy is: ${((metrics.accuracy * 100) * 100).round / 100.toDouble}%")
    val t5 = System.nanoTime()
    println("Elapsed time: " + ((t5 - t4) / 1E9).toInt + " seconds")
  }
}
