import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object LinkPredictionUnsupervised {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Link Prediction Unsupervised").master("local[*]").getOrCreate()

    //Read files
    val pairs_df = ss.read
      .option("header", value = true)
      .option("delimiter", value = ",")
      .option("inferSchema", "true")
      .csv("src/main/resources/pairs_with_features.csv")


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

    val pairData = assembler
      .transform(pairs_df)
      .cache()

    println("Starting model training")
    val t0 = System.nanoTime()

    // Trains a k-means model.
    // val kmeans = new KMeans().setK(2).setSeed(1L).setTol(1E-9).setMaxIter(1000)
    // val model = kmeans.fit(pairData)
    // model.write.overwrite().save("src/main/models/KMeans")
    val model = KMeansModel.load("src/main/models/KMeans")

    // Trains a bisecting k-means model.
    // val bkm = new BisectingKMeans().setK(2).setSeed(1L).setMaxIter(50)
    // val model = bkm.fit(pairData)
    // model.write.overwrite().save("src/main/models/BKMeans")
    // val model = BisectingKMeansModel.load("src/main/models/BKMeans")

    // Trains Gaussian Mixture Model
    // val gmm = new GaussianMixture().setK(2).setSeed(1L).setTol(1E-9).setMaxIter(1000)
    // val model = gmm.fit(pairData)
    // model.write.overwrite().save("src/main/models/GMM")
    // val model = GaussianMixtureModel.load("src/main/models/GMM")

    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1E9).toInt + " seconds")
    println("Starting model Prediction")
    val t2 = System.nanoTime()

    // Predict
    val predictions = model.transform(pairData)

    val scoreAndLabels = predictions
      .select("label", "prediction")
      .rdd
      .map(p => (p.getInt(0).toDouble, p.getInt(1).toDouble))
      .cache()

    val metrics = new MulticlassMetrics(scoreAndLabels)
    println(s"Weighted precision: ${((metrics.weightedPrecision * 100) * 100).round / 100.toDouble}%")
    println(s"Weighted recall: ${((metrics.weightedRecall * 100) * 100).round / 100.toDouble}%")
    println(s"Weighted F1 score: ${((metrics.weightedFMeasure * 100) * 100).round / 100.toDouble}%")
    println(s"Precision of citation: ${((metrics.recall(1.0) * 100) * 100).round / 100.toDouble}%") // Yes, MulticlassMetrics has a bug
    println(s"Accuracy: ${((metrics.accuracy * 100) * 100).round / 100.toDouble}%")
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    val t3 = System.nanoTime()
    println("Elapsed time: " + ((t3 - t2) / 1E9).toInt + " seconds")

  }
}
