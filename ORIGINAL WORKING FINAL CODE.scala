import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.classification.LogisticRegression 
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator 
import org.apache.spark.ml.evaluation.RegressionEvaluator 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.Pipeline 
import org.apache.spark.ml.param.ParamMap 
import org.apache.spark.ml.param._
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor} 
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor 
import org.apache.spark.ml.tuning.CrossValidator 
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row


val traindatatmp1= spark.read.format("csv").option( "header", "true").option("inferSchema","true").load("/FileStore/tables/titanicdata/train.csv")

//Finding null count in each column
val col=traindatatmp1.columns
var dfArray=col.map(colmn=>traindatatmp1.select(lit(colmn).as("colName"),sum(when(traindatatmp1(colmn).isNull || traindatatmp1(colmn)==="" || traindatatmp1(colmn)==="-" || traindatatmp1(colmn).isNaN,1).otherwise(0)).as("missingValues")))
dfArray.tail.foldLeft(dfArray.head)((acc,itr)=>acc.union(itr)).show

val traindatatmp=traindatatmp1.drop("Cabin","Ticket","Name")

val maledata= traindatatmp.filter("Sex= 'male'").select("PassengerId","Age")
val maledata2 = maledata.filter("Age IS NOT NULL")
var totMAge =  maledata2.agg(sum("Age")).first.get(0)
val totMAge2:Long = totMAge.asInstanceOf[Number].longValue
var cntM= maledata2.count()
val avgMage=totMAge2/cntM
val maledata3 = maledata.na.fill(avgMage)

val femaledata= traindatatmp.filter("Sex= 'female'").select("PassengerId","Age")
val femaledata2 = femaledata.filter("Age IS NOT NULL")
var totFAge =  femaledata2.agg(sum("Age")).first.get(0)
val totFAge2:Long = totFAge.asInstanceOf[Number].longValue
val cntF=femaledata2.count()
val avgFage=totFAge2/cntF
val femaledata3 = femaledata.na.fill(avgFage)

val agedata=maledata3.union(femaledata3).sort($"PassengerId").as('ag)

val traindatatmp2= traindatatmp.drop(traindatatmp.col("Age")).as('tr)

val traindata= traindatatmp2
  .join(agedata, $"tr.PassengerId" === $"ag.PassengerId")
  .select($"tr.PassengerId" as "PassengerId",$"tr.Survived" as "Survived",$"tr.Pclass" as "Pclass",$"tr.Sex" as "Sex",$"ag.Age" as "Age",$"tr.SibSp" as "SibSp", $"tr.Parch" as "Parch",$"tr.Fare" as "Fare",$"tr.Embarked" as "Embarked")

traindata.show

traindata.select("Age","SibSp","Parch","Fare").describe().show()

traindata.groupBy("Sex").agg(count("PassengerId"),min("Age"),max("Age"), min("Fare"),max("Fare")).show

traindata.groupBy("Embarked","Sex").agg(count("PassengerId")).show

val td= traindata.withColumn( "PassengerId", traindata.col( "PassengerId").cast(StringType)); 

//Correlation Matrix
// All the columns which are continuous is aggregated to a column called "features_corr" for finding the correlation matrix
val assembler_corr = new VectorAssembler().setInputCols(Array( "Age","SibSp","Parch","Fare")).setOutputCol("features_corr")
assembler_corr.setHandleInvalid("skip").transform(traindata).show
val pipeline_corr =new Pipeline().setStages(Array(assembler_corr))
val dffull_corr = pipeline_corr.fit(traindata).transform(traindata)

val Row(coeff1: Matrix) = Correlation.corr(dffull_corr, "features_corr").head
println("Pearson correlation matrix:\n|-----------Age-------|--------SibSp-------|---------Parch-------|----------Fare---------| \n" + coeff1.toString)

// Categorical Variables 
val categCol = Array("Sex" , "Pclass", "Embarked")

var transTD = td;
for (c <- categCol) 
{
	val str1 = c+ "_Index";
	val str2 = c+ "_Vec"; 
	val indexer = new StringIndexer().setInputCol(c).setOutputCol(str1)
	val encoder = new OneHotEncoder().setInputCol(str1).setOutputCol(str2)
	val pipeline = new Pipeline().setStages(Array(indexer, encoder))
	transTD = pipeline.fit(transTD).transform(transTD); 
}
transTD.show()

// All the columns we are intersEed in should be aggregated eo a column called "features"
val assembler = new VectorAssembler().setInputCols(Array( "Age","SibSp","Parch","Fare","Sex_Vec","Pclass_Vec","Embarked_Vec")).setOutputCol("features")
assembler.setHandleInvalid("skip").transform(transTD).show
val pipeline =new Pipeline().setStages(Array(assembler))
val dffull = pipeline.fit(transTD).transform(transTD)

// Split data into test and train 
val Array(training, test) = dffull.randomSplit(Array(0.7, 0.3), 18)
training.printSchema 
test.show

// Create two Evaluators
val binaryClassificationEvaluator= new BinaryClassificationEvaluator().setLabelCol("Survived").setRawPredictionCol("rawPrediction")
binaryClassificationEvaluator.setMetricName("areaUnderROC") 
val regressionEvaluator = new RegressionEvaluator().setLabelCol("Survived").setPredictionCol("prediction") 
regressionEvaluator.setMetricName("rmse") 

// Logistic Regression 
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFeaturesCol("features").setLabelCol("Survived") 
val lrModel = lr.fit(training) 
val predictions = lrModel.transform(training)
val areaTraining = binaryClassificationEvaluator.evaluate(predictions) 
println( "Area under ROC using Logistic Regression on training data = " + areaTraining) 
val predictionsTest = lrModel.transform(test) 
val areaTest = binaryClassificationEvaluator.evaluate(predictionsTest)
println("Area under ROC using Logistic Regression on test data = "+ areaTest ) 
val rmseLR = regressionEvaluator.evaluate(predictionsTest)
println("Root Mean Squared Error (RMSE) Logistic Regression on test data = "+ rmseLR) 


//Crossvalidator

val pipelineCV = new Pipeline().setStages(Array(lr))
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array (0.05, 0.1,0.2)).addGrid (lr.maxIter, Array(5, 10, 15)).build 
val cv = new CrossValidator().setNumFolds(10).setEstimator(pipelineCV).setEstimatorParamMaps(paramGrid).setEvaluator(binaryClassificationEvaluator)
val cvModel = cv.fit(training)
val predictionsCvTest = cvModel.transform(test)
val areaLrCvTest = binaryClassificationEvaluator.evaluate(predictionsCvTest) 
println( "Area under ROC using Logistic Regression with Weight Column on test data = "+ areaLrCvTest) 
val rmseCv = regressionEvaluator.evaluate(predictionsCvTest)
println( "Root Mean Squared Error (RMSE) Logistic Regression with Weight Column on test data = " + rmseCv)


// Decision Trees 

val maxTreeDepth=5
val dt= new DecisionTreeRegressor().setLabelCol("Survived").setFeaturesCol("features").setMaxBins(32).setMaxDepth(5)
val pipelineDT= new Pipeline().setStages(Array(dt))
val modelDT = pipelineDT.fit(training)
val predictionsDT= modelDT.transform(test)
val rmseDT= regressionEvaluator.evaluate(predictionsDT)
println( "Root Mean Squared Error (RMSE) using Decision Trees on test data =" + rmseDT)

// Random Forest

val rf= new RandomForestRegressor().setLabelCol("Survived").setFeaturesCol("features")
val pipelineRF= new Pipeline().setStages(Array(rf))
val modelRF = pipelineRF.fit(training)
val predictionsRF= modelRF.transform(test)
val rmseRF= regressionEvaluator.evaluate(predictionsRF)
println( "Root Mean Squared Error (RMSE) using Random Forest on test data =" + rmseRF)

// Gradient Boosting

val gbt= new GBTRegressor().setLabelCol("Survived").setFeaturesCol("features").setMaxIter(10)
val pipelineGBT= new Pipeline().setStages(Array(gbt))
val modelGBT = pipelineGBT.fit(training)
val predictionsGBT= modelGBT.transform(test)
val rmseGBT= regressionEvaluator.evaluate(predictionsGBT)
println( "Root Mean Squared Error (RMSE) using Gradient Boosting on test data =" + rmseGBT)