import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel, GaussianMixture, KMeans}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object LinkPredictionUnsupervised {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val NUM_TO_TEST = 1000

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Link Prediction Unsupervised").master("local[*]").getOrCreate()
    import ss.implicits._

    //Read files
    val training_set_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = ",")
      .option("inferSchema", "true")
      .csv("src/main/resources/training_set_with_features.csv")

    val testing_set_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = ",")
      .option("inferSchema", "true")
      .csv("src/main/resources/testing_set_with_features.csv")

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "num_of_same_words_in_title",
        "num_of_same_words_in_abstract",
        "cosine_similarity",
        "have_same_authors",
        "year_gap"
      ))
      .setOutputCol("features")
      .setHandleInvalid("skip")


    val ground_truth_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = " ")
      .option("inferSchema", "true")
      .csv("src/main/resources/Cit-HepTh.txt")

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

    val trainingData = assembler
      .transform(testing_set_df
        .withColumn("label", getLabel($"Target", $"Source"))
      )
      .select("features")
      .cache()
    val testData = assembler
      .transform(testing_set_df
        .withColumn("label", getLabel($"Target", $"Source"))
        .limit(NUM_TO_TEST))
      .select("label", "features")


    println("Starting model training")
    val t0 = System.nanoTime()

    // Trains a k-means model.
    // val kmeans = new KMeans().setK(2).setSeed(1L).setTol(1E-9).setMaxIter(1000)
    // val model = kmeans.fit(trainingData)


    // Trains a bisecting k-means model.
    val bkm = new BisectingKMeans().setK(2).setSeed(1L).setMaxIter(50)
    val model = bkm.fit(trainingData)
    model.write.overwrite().save("src/main/models/BKMeans")
    //val model = BisectingKMeansModel.load("src/main/models/BKMeans")

    // Trains Gaussian Mixture Model
    // val gmm = new GaussianMixture().setK(2).setSeed(1L).setTol(1E-9).setMaxIter(1000)
    // val model = gmm.fit(trainingData)

    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1E9).toInt + " seconds")
    println("Starting model Prediction")
    val t2 = System.nanoTime()

    // Predict
    val predictions = model.transform(testData)
    val scoreAndLabels = predictions.select("label", "prediction")
      .rdd
      .map(p => (p.getInt(0).toDouble, p.getInt(1).toDouble))
      .cache()
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
    val f1 = evaluator.evaluate(scoreAndLabels.toDF("label", "prediction"))
    val metrics = new MulticlassMetrics(scoreAndLabels)
    println(metrics.confusionMatrix)
    println(s"F1 Score is: ${((f1 * 100) * 100).round / 100.toDouble}%")

    val t3 = System.nanoTime()
    println("Elapsed time: " + ((t3 - t2) / 1E9).toInt + " seconds")

  }
}