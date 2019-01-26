package outlier.detection

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD

class AnomalyDetection extends Serializable with Logging {

  val default_epsilon: Double = 0.003

  def run(data: RDD[Vector]): AnomalyDetectionModel = {

    val stats: MultivariateStatisticalSummary = Statistics.colStats(data)
    val mean: Vector = stats.mean
    val variances: Vector = stats.variance
    logInfo("MEAN %s VARIANCE %s".format(mean, variances))

    new AnomalyDetectionModel(mean, variances, default_epsilon)
  }

}


object AnomalyDetection {

  private[detection] def predict(point: Vector, means: Vector, variances: Vector, epsilon: Double): Boolean = {
    probFunction(point, means, variances) < epsilon
  }

  private[detection] def probFunction(point: Vector, means: Vector, variances: Vector): Double = {
    val tripletByFeature: List[(Double, Double, Double)] = (point.toArray, means.toArray, variances.toArray).zipped.toList
    tripletByFeature.map { triplet =>
      val x = triplet._1
      val mean = triplet._2
      val variance = triplet._3
      val expValue = Math.pow(Math.E, -0.5 * Math.pow(x - mean, 2) / variance)
      (1.0 / (Math.sqrt(variance) * Math.sqrt(2.0 * Math.PI))) * expValue
    }.product
  }


}



