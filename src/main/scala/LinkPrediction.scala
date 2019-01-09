import org.apache.hadoop.yarn.util.RackResolver
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object LinkPrediction {

  Logger.getLogger(classOf[RackResolver]).getLevel
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val NUM_TO_TEST = 1000

    //Create a SparkSession to initialize Spark
    val ss = SparkSession.builder().appName("Link Prediction").master("local[*]").getOrCreate()
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
        "Source",
        "Target",
        "num_of_same_words_in_title",
        "num_of_same_words_in_abstract",
        "source_indegree",
        "target_indegree",
        "source_outdegree",
        "target_outdegree",
        "cosine_similarity",
        "neighbour_similarity",
        "have_same_authors",
        "year_gap"
      ))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val trainingData = assembler
      .transform(training_set_df.withColumnRenamed("Edge", "label"))
      .select("label", "features")
      .cache()

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

    val testData = assembler
      .transform(testing_set_df
        .withColumn("label", getLabel($"Target", $"Source"))
        .limit(NUM_TO_TEST))
      .select("label", "features")

    println("Starting model training")
    val t0 = System.nanoTime()

    // Train LSVC model
    // val model = new LinearSVC().setMaxIter(2000).setRegParam(0.001).setTol(1E-8).setStandardization(false).setAggregationDepth(10).fit(trainingData)
    // model.write.overwrite().save("src/main/models/LSVC")
    // val modelLSVC = LinearSVCModel.load("src/main/models/LSVC")

    // Train Logistic Regression model
    // val model = new LogisticRegression().setMaxIter(2000).setTol(1E-8).fit(trainingData)
    // model.write.overwrite().save("src/main/models/LogisticRegression")
    // val modelLogReg = LogisticRegressionModel.load("src/main/models/LogisticRegression")

    // Train Neural Network
    // val model = new MultilayerPerceptronClassifier().setLayers(Array[Int](10, 32, 64, 2)).setMaxIter(1000).fit(trainingData)
    // model.write.overwrite().save("src/main/models/NeuralNetwork")
    // val modelNN = MultilayerPerceptronClassificationModel.load("src/main/models/NeuralNetwork")

    // Train Ramdom Forest
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(trainingData)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(trainingData)
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(70).setMaxDepth(12)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val modelRandomForest = pipeline.fit(trainingData)
    modelRandomForest.write.overwrite().save("src/main/models/RandomForest")

    // Train GBoost Tree
    // val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(1000).setFeatureSubsetStrategy("auto")
    // val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
    // val model = pipeline.fit(trainingData)

    val t1 = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1E9).toInt + " seconds")
    println("Starting model Prediction")
    val t2 = System.nanoTime()

    val model = modelRandomForest
    val predictions = model.transform(testData)

    // Predict (if not trees)
    // val scoreAndLabels = predictions.select("prediction", "label")
    //  .rdd
    //  .map(p => (p.getDouble(0), p.getInt(1).toDouble))
    //  .cache()

    //    val metrics = new MulticlassMetrics(scoreAndLabels)
    //    println(metrics.confusionMatrix)

    // Predict (if trees)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("f1")
    val f1 = evaluator.evaluate(predictions)

    println(s"F1 Score is: ${((f1 * 100) * 100).round / 100.toDouble}%")
    val t3 = System.nanoTime()
    println("Elapsed time: " + ((t3 - t2) / 1E9).toInt + " seconds")

  }
}