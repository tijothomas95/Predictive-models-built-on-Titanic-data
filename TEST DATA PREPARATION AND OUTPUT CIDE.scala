val testdatatmp1= spark.read.format("csv").option( "header", "true").option("inferSchema","true").load("/FileStore/tables/titanicdata/test.csv")

//Finding null count in each column
val tcol=testdatatmp1.columns
var tdfArray=tcol.map(colmn=>testdatatmp1.select(lit(colmn).as("colName"),sum(when(testdatatmp1(colmn).isNull || testdatatmp1(colmn)==="" || testdatatmp1(colmn)==="-" || testdatatmp1(colmn).isNaN,1).otherwise(0)).as("missingValues")))
tdfArray.tail.foldLeft(tdfArray.head)((acc,itr)=>acc.union(itr)).show

val testdatatmp=testdatatmp1.drop("Cabin","Ticket","Name")

val tmaledata= testdatatmp.filter("Sex= 'male'").select("PassengerId","Age")
val tmaledata2 = tmaledata.filter("Age IS NOT NULL")
var ttotMAge =  tmaledata2.agg(sum("Age")).first.get(0)
val ttotMAge2:Long = ttotMAge.asInstanceOf[Number].longValue
var tcntM= tmaledata2.count()
val tavgMage=ttotMAge2/tcntM
val tmaledata3 = tmaledata.na.fill(tavgMage)

val tfemaledata= testdatatmp.filter("Sex= 'female'").select("PassengerId","Age")
val tfemaledata2 = tfemaledata.filter("Age IS NOT NULL")
var ttotFAge =  tfemaledata2.agg(sum("Age")).first.get(0)
val ttotFAge2:Long = ttotFAge.asInstanceOf[Number].longValue
val tcntF=tfemaledata2.count()
val tavgFage=ttotFAge2/tcntF
val tfemaledata3 = tfemaledata.na.fill(tavgFage)

val tagedata=tmaledata3.union(tfemaledata3).sort($"PassengerId").as('tag)

val testdatatmp2= testdatatmp.drop(testdatatmp.col("Age")).as('ttr)

val testdata= testdatatmp2
  .join(tagedata, $"ttr.PassengerId" === $"tag.PassengerId")
  .select($"ttr.PassengerId" as "PassengerId",$"ttr.Pclass" as "Pclass",$"ttr.Sex" as "Sex",$"tag.Age" as "Age",$"ttr.SibSp" as "SibSp", $"ttr.Parch" as "Parch",$"ttr.Fare" as "Fare",$"ttr.Embarked" as "Embarked")

val ttd= testdata.withColumn( "PassengerId", testdata.col( "PassengerId").cast(StringType)); 

// Categorical Variables 
val tcategCol = Array("Sex" , "Pclass", "Embarked")

var ttransTD = ttd;
for (tc <- tcategCol) 
{
  val tstr1 = tc+ "_Index";
  val tstr2 = tc+ "_Vec"; 
  val tindexer = new StringIndexer().setInputCol(tc).setOutputCol(tstr1)
  val tencoder = new OneHotEncoder().setInputCol(tstr1).setOutputCol(tstr2)
  val tpipeline = new Pipeline().setStages(Array(tindexer, tencoder))
  ttransTD = tpipeline.fit(ttransTD).transform(ttransTD); 
}

// All the columns we are intersEed in should be aggregated eo a column called "features"
val tassembler = new VectorAssembler().setInputCols(Array( "Age","SibSp","Parch","Fare","Sex_Vec","Pclass_Vec","Embarked_Vec")).setOutputCol("features")
tassembler.setHandleInvalid("skip").transform(ttransTD).show
val tpipeline =new Pipeline().setStages(Array(tassembler))
val tdffull = pipeline.fit(ttransTD).transform(ttransTD)

val tpredictionsTestLR = lrModel.transform(tdffull)
tpredictionsTestLR.select("prediction").write.format("csv").save("/FileStore/tables/LRouttestt")

val tpredictionsTestCV = cvModel.transform(tdffull)
tpredictionsTestCV.select("prediction").write.format("csv").save("/FileStore/tables/CVouttestt")

val tpredictionsTestDT = modelDT.transform(tdffull)
tpredictionsTestDT.select("prediction").write.format("csv").save("/FileStore/tables/DTouttestt")

val tpredictionsTestRF = modelRF.transform(tdffull)
tpredictionsTestRF.select("prediction").write.format("csv").save("/FileStore/tables/RFouttestt")

val tpredictionsTestGBT = modelGBT.transform(tdffull)
tpredictionsTestGBT.select("prediction").write.format("csv").save("/FileStore/tables/GBTouttestt")