import java.io.ObjectInputStream

import com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier
import com.hankcs.hanlp.classification.corpus.MemoryDataSet
import com.hankcs.hanlp.classification.models.NaiveBayesModel
import org.apache.commons.lang3.SerializationUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType


object BayesTrainer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("BAYES-TRAINER").enableHiveSupport().getOrCreate()
    val trainDF = spark.emptyDataFrame.select("id", "text")
    val memoryDataSet = new MemoryDataSet
    trainDF
      .collect()
      .foreach(row => memoryDataSet.add(row.getString(0), row.getString(1)))

    val classifier = new NaiveBayesClassifier
    classifier.train(memoryDataSet)
    val classifierModel = classifier.getModel.asInstanceOf[NaiveBayesModel]

    val classifyUDF: UserDefinedFunction = udf((text: String) => new NaiveBayesClassifier(classifierModel).classify(text), StringType)

    val modelPath = new Path("")
    val fileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    if (fileSystem.exists(modelPath)) {
      fileSystem.delete(modelPath)
    }
    val outputStream = fileSystem.create(modelPath)
    SerializationUtils.serialize(classifierModel, outputStream)

    val ois = new ObjectInputStream(fileSystem.open(modelPath))
    val loadedClassifyModel = ois.readObject().asInstanceOf[NaiveBayesModel]
    ois.close()
    val loadedClassifier = new NaiveBayesClassifier(loadedClassifyModel)
    val testText = "今日头条，今天北京大到暴雨"
    println(s"$testText 的分类是 ${loadedClassifier.classify(testText)}")

    spark.stop()
  }

}
