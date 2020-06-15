package transformer

import java.io.ObjectInputStream

import com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier
import com.hankcs.hanlp.classification.models.NaiveBayesModel
import com.sohu.tv.push.ClassifierTrainer.MODEL_PATH
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

class HanLPClassifier(override val uid: String)
  extends UnaryTransformer[String, String, HanLPClassifier] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("hanLPClassifier"))
  }

  override def createTransformFunc: String => String = { originStr =>
    val classifier = new NaiveBayesClassifier(loadModel())
    val mClass = classifier.classify(originStr)
    mClass
  }

  def loadModel() : NaiveBayesModel = {
    val sc = SparkSession.builder().getOrCreate()
    val modelPath = new Path(MODEL_PATH)
    val fileSystem = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    val ois = new ObjectInputStream(fileSystem.open(modelPath))
    val model = ois.readObject().asInstanceOf[NaiveBayesModel]
    model
  }

  override def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType, s"Input type must be string type but got $inputType.")
  }

  override def outputDataType: DataType = {
    StringType
  }

  override def copy(extra: ParamMap): HanLPClassifier = {
    defaultCopy(extra)
  }
}


