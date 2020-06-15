import Conf.STOPWORD_ARRAY
import org.apache.spark.ml.feature.{StopWordsRemover, Word2Vec, Word2VecModel}
import org.apache.spark.sql.SparkSession
import transformer.HanLPTokenizer

/**
  * Author: sunyi
  * Date: 2020/6/15 14:14
  * Version 1.0
  */
object Word2vecTrainer {
  val spark = SparkSession.builder().appName("TV-PUSH-TRAIN-WORD2VEC").enableHiveSupport().getOrCreate()

  import spark.implicits._

  val trainDF = spark.emptyDataFrame.select("text")
  val columnName = "text"
  val hanLPTokenizer = new HanLPTokenizer()
    .setInputCol(columnName)
    .setOutputCol(s"${columnName}_words")
    .setShouldRemoveStopWords(true)
  val tokenizedDF = hanLPTokenizer.transform(trainDF)

  val stopWordsRemover = new StopWordsRemover()
    .setInputCol(s"${columnName}_words")
    .setOutputCol(s"${columnName}_filtered_words")
    .setStopWords(STOPWORD_ARRAY)
  val filteredDF = stopWordsRemover.transform(tokenizedDF)

  val word2Vec = new Word2Vec()
    .setInputCol(s"${columnName}_filtered_words")
    .setOutputCol(s"${columnName}_w2v")
    .setMaxIter(30)
    .setVectorSize(200)
    .setWindowSize(5)
    .setMinCount(10)
  val word2VecModel = word2Vec.fit(trainDF)
  word2VecModel.save("") // 自己训练的实例模型保存模型
  val mode: Word2VecModel = Word2VecModel.load("") //Word2Vec半生对象加载模型
  println(word2VecModel.explainParams())

  val word2VecDF = word2VecModel.transform(trainDF)
  word2VecDF.show(false)

  spark.stop()
}
