import org.apache.spark.ml.clustering.{LDA, LocalLDAModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover}
import org.apache.spark.sql.SparkSession
import transformer.HanLPTokenizer

/**
  * Author: sunyi
  * Date: 2020/6/15 14:30
  * Version 1.0
  */
object LDATrainer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("LDA-TRAINER").enableHiveSupport().getOrCreate()
    val df = spark.emptyDataFrame.select("text")
    val hanLPTokenizer = new HanLPTokenizer()
      .setInputCol("text")
      .setOutputCol(s"text_words")
      .setShouldRemoveStopWords(true)
    val tokenizedDF = hanLPTokenizer.transform(df)
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(s"text_words")
      .setOutputCol(s"text_filtered_words")
      .setStopWords(Conf.STOPWORD_ARRAY)
    val filteredDF = stopWordsRemover.transform(tokenizedDF)

    val tfDF = new HashingTF()
      .setInputCol("text_filtered_words")
      .setOutputCol("tf_features")
      .transform(filteredDF)

    val idf = new IDF()
      .setInputCol("tf_features")
      .setOutputCol("features")
      .setMinDocFreq(2)
    val idfModel = idf.fit(tfDF)

    val idfDF = idfModel.transform(tfDF)
      .cache()

    val lda = new LDA()
      .setMaxIter(50)
      .setK(10)
      .setFeaturesCol("features")
    val ldaModel = lda.fit(idfDF).asInstanceOf[LocalLDAModel]

    val ldaData = ldaModel.transform(idfDF)
    val ll = ldaModel.logLikelihood(idfDF)
    val lp = ldaModel.logPerplexity(idfDF)
    println(s"Likelihood: $ll")
    println(s"Perplexity: $lp")
    val topics = ldaModel.describeTopics(maxTermsPerTopic = 3)
    println("The topics described by their top-weighted terms:")
    topics.show(false)


    spark.stop()
  }
}
