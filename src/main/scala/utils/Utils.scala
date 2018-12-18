package utils

import org.apache.spark.sql.SparkSession

import scala.collection.mutable

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

  def cosineSimilarity(x: Array[Int], y: Array[Int]): Double = {
    require(x.length == y.length)
    dotProduct(x, y) / (magnitude(x) * magnitude(y))
  }

  def dotProduct(x: Array[Int], y: Array[Int]): Int = {
    (for ((a, b) <- x zip y) yield a * b) sum
  }

  def magnitude(x: Array[Int]): Double = {
    math.sqrt(x map (i => i * i) sum)
  }


  def similarity(t1: Map[String, Int], t2: Map[String, Int]): Double = {

    val m = mutable.HashMap[String, (Int, Int)]()

    val sum1 = t1.foldLeft(0d) {
      case (sum, (word, freq)) =>
        m += word -> (freq, 0)
        sum + freq
    }

    val sum2 = t2.foldLeft(0d) {
      case (sum, (word, freq)) =>
        m.get(word) match {
          case Some((freq1, _)) => m += word -> (freq1, freq)
          case None => m += word -> (0, freq)
        }
        sum + freq
    }

    val (p1, p2, p3) = m.foldLeft((0d, 0d, 0d)) {
      case ((s1, s2, s3), e) =>
        val fs = e._2
        val f1 = fs._1 / sum1
        val f2 = fs._2 / sum2
        (s1 + f1 * f2, s2 + f1 * f1, s3 + f2 * f2)
    }

    val cos = p1 / (Math.sqrt(p2) * Math.sqrt(p3))
    cos
  }
}
