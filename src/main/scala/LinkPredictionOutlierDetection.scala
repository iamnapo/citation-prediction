import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import outlier.detection.AnomalyDetection

object LinkPredictionOutlierDetection {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Link Prediction Outlierr Detection").master("local[*]").getOrCreate()
    val sc = ss.sparkContext

    //Read files
    val pairVectors: RDD[mllib.linalg.Vector] = sc.textFile("src/main/resources/pairs_outliers.txt", 2)
      .map(_.split(",").map(_.toDouble))
      .map(arrDouble => Vectors.dense(arrDouble))


    val cvVectors = sc.textFile("src/main/resources/pairs_cross_validation.txt", 2).cache()
      .map(_.split(",").map(_.toDouble))
      .map(arrDouble => new LabeledPoint(arrDouble(0), Vectors.dense(arrDouble.slice(1, arrDouble.length))))


    println("Starting model training")
    val t0 = System.nanoTime()

    val pairs = pairVectors.cache()
    val anDet: AnomalyDetection = new AnomalyDetection()
    val model = anDet.run(pairs)
    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1E9).toInt + " seconds")
    println("Starting model Prediction")
    val t2 = System.nanoTime()

    // Predict

    val results = model.predict(cvVectors)

    val scoreAndLabels = results
      .map(row => (row._3, if (row._2) 1.0 else 0.0))
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
