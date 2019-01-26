package outlier.detection

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class AnomalyDetectionModel (val means: Vector, val variances: Vector, val epsilon: Double) extends Serializable {

  def predict(point: Vector): Boolean = {
    AnomalyDetection.predict(point, means, variances, epsilon)
  }

  def predict(points: RDD[LabeledPoint]): RDD[(Vector, Boolean, Double)] = {
    points.map(p => (p.features,AnomalyDetection.predict(p.features, means, variances, epsilon), p.label))
  }

}