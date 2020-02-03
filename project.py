from pyspark.sql import functions as F
sc.install_pypi_package("pandas==0.25.1")
sc.install_pypi_package("matplotlib", "https://pypi.org/simple")
sc.install_pypi_package("s3fs==0.3.5")
sc.install_pypi_package("scikit-learn==0.21.3")

import pandas as pd
input_bucket = 's3://bdprojtoxic'
input_path = '/train.csv'
df = pd.read_csv(input_bucket + input_path)

c = 0
for i in df["toxic"]:
  if(i == 1):
    c += 1

print("Distribution of Comments ")
print("Toxic : " , (c/len(df))*100, "%")

c = 0
for i in df["severe_toxic"]:
  if(i == 1):
    c += 1
    
print("Severe Toxic : " , (c/len(df))*100, "%")

c = 0
for i in df["obscene"]:
  if(i == 1):
    c += 1
    
print("Obscene : " , (c/len(df))*100, "%")
    
c = 0
for i in df["threat"]:
  if(i == 1):
    c += 1
    
print("Threat : " , (c/len(df))*100, "%")
  
c = 0
for i in df["insult"]:
  if(i == 1):
    c += 1
    
print("Insult : " , (c/len(df))*100, "%")
    
c = 0
for i in df["identity_hate"]:
  if(i == 1):
    c += 1
    
print("Identity Hate : " , (c/len(df))*100, "%")

from pyspark.ml.feature import StopWordsRemover
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
sdf = spark.createDataFrame(df)
sdf.show()

snewdf = sdf.withColumn("comment_text" , F.regexp_replace(F.col("comment_text"), "[\$#,?.""!@#$%^&*()0123456789:-=\+]", ""))
snewdf = snewdf.withColumn('comment_text', F.regexp_replace(F.col("comment_text"), "\"", ""))
snewdf = snewdf.withColumn('comment_text', F.regexp_replace(F.col("comment_text"), "\n", " "))
snewdf = snewdf.withColumn('comment_text', F.regexp_replace(F.col("comment_text"), "\[", ""))
snewdf = snewdf.withColumn('comment_text', F.regexp_replace(F.col("comment_text"), "\]", ""))
snewdf = snewdf.withColumn('comment_text', F.regexp_replace(F.col("comment_text"), "\"+", ""))
snewdf = snewdf.withColumn('comment_text', F.lower(F.col('comment_text')))
snewdf  = snewdf.withColumn('comment_text', F.rtrim(snewdf.comment_text))
snewdf  = snewdf.withColumn('comment_text', F.ltrim(snewdf.comment_text))

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol="comment_text", outputCol="tokenized")
tokenized_df = tokenizer.transform(snewdf)
tokenized_df.select("tokenized").show()

stopwordsremoved = StopWordsRemover(inputCol="tokenized", outputCol="comment_txt")
swr_df = stopwordsremoved.transform(tokenized_df)
swr_df.select("comment_txt").show()

from pyspark.ml.feature import HashingTF, IDF

hashingTF = HashingTF().setNumFeatures(50).setInputCol("comment_txt").setOutputCol("features")

newdf = hashingTF.transform(swr_df)

idf = IDF(inputCol="features", outputCol="final_features")
idfModel = idf.fit(newdf) 
tfidf = idfModel.transform(newdf)

from pyspark.sql import Row
f_features = tfidf.select("final_features")

pDf = f_features.toPandas()
pDf.info()

import numpy as np
newpdf = pDf['final_features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)

features = np.apply_along_axis(lambda x : x[0], 1, newpdf)

allLabels = tfidf.select(["toxic", "severe_toxic", "obscene","threat","identity_hate", "insult"]).toPandas()
print(allLabels)

allarr = allLabels.apply(lambda x : np.array(x)).as_matrix().reshape(-1,6)
print(allarr.shape)
print(allarr[6])

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(features, allarr, test_size=0.3, random_state=4)

test_y[:,1].reshape(len(train_y), 1).shape

from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
output = []

class NeuralNet:
    def __init__(self, label, header = None, h1 = 9, h2 = 3):
        np.random.seed(1)
        
        self.X = train_x
        self.y = train_y[:,label].reshape(len(train_y), 1)
        
        input_layer_size = len(self.X[0])
        output_layer_size = 1
    
        self.test = test_x
        self.testy = test_y[:,label].reshape(len(test_y), 1)

    
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    
    
    

    def train(self, activation, max_iterations = 100, lr = 0.1):
        
        out = None
        for iteration in range(max_iterations):

            if activation == "sigmoid":
                in1 = np.dot(self.X, self.w01 )
                self.X12 = 1 / (1 + np.exp(-in1))
                
                in2 = np.dot(self.X12, self.w12)
                self.X23 = 1 / (1 + np.exp(-in2))
                
                in3 = np.dot(self.X23, self.w23)
                out = 1 / (1 + np.exp(-in3))
                
                error = 0.5 * np.power((out - self.y), 2)
            
            
                self.deltaOut = (self.y - out) * (out * (1 - out))

                self.delta23 = (self.deltaOut.dot(self.w23.T)) * (self.X23 * (1 - self.X23))

                self.delta12 = (self.delta23.dot(self.w12.T)) * (self.X12 * (1 - self.X12))
            
            if activation == "relu":
                in1 = np.dot(self.X, self.w01 )
                self.X12 = in1 * (in1 > 0)
                
                in2 = np.dot(self.X12, self.w12)
                self.X23 = in2 * (in2 > 0)
                
                in3 = np.dot(self.X23, self.w23)
                out = in3 * (in3 > 0)

                error = 0.5 * np.power((out - self.y), 2)
                
                self.deltaOut = (self.y - out) * (1 * (out > 0))
                
                self.delta23 = (self.deltaOut.dot(self.w23.T)) * (1 * (self.X23 > 0))

                self.delta12 = (self.delta23.dot(self.w12.T)) * (1 * (self.X12 > 0))
        

            self.w23 += lr * self.X23.T.dot(self.deltaOut)
            self.w12 += lr * self.X12.T.dot(self.delta23)
            self.w01 += lr * self.X01.T.dot(self.delta12)

        print ("The activation function used is: " + activation)
        print ("The learning rate is: " + str(lr))    
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")



    def predict(self, activation):
        if activation == "sigmoid":
            in1 = np.dot(self.test, self.w01 )
            self.X12 = 1 / (1 + np.exp(-in1))
            in2 = np.dot(self.X12, self.w12)
            self.X23 = 1 / (1 + np.exp(-in2))
            in3 = np.dot(self.X23, self.w23)
            out = 1 / (1 + np.exp(-in3))
            
        elif activation == "relu":
            in1 = np.dot(self.test, self.w01 )
            self.X12 = in1 * (in1 > 0)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = in2 * (in2 > 0)
            in3 = np.dot(self.X23, self.w23)
            out = in3 * (in3 > 0)
            
        
        error = 0.5 * np.power((out - self.testy), 2)
        print("The test error:"+ str(np.sum(error)))
        
        out_max = max(out)
        out_min = min(out)
        new_out = [0]*len(out)
        
        for i in range(0, len(out)):
            new_out[i] = (out[i] - out_min) / (out_max - out_min)
        
        third_q = np.percentile(new_out, 75)
        
#         print("Q2 quantile of arr : ", np.percentile(new_out, 50)) 
#         print("Q1 quantile of arr : ", np.percentile(new_out, 25)) 
        print("Q3 quantile of arr : ", third_q) 
#         print("100th quantile of arr : ", np.percentile(new_out, 100))
        
        count = 0
        l = []
        for i in range(0, len(out)):
            if (out[i] <= third_q):
                if (self.testy[i] != 0):
                    count+=1
                l.append(0)
            else:
                if (self.testy[i] != 1):
                    count+=1
                l.append(1)
        
        print("Accuracy: " , ((len(out)-count)/len(out))*100);
        print("Misclassify: " + str(count)); 
        return l

lis = []
for i in range(0, 6):
    neural_network = NeuralNet(i)
    neural_network.train("sigmoid")
    lis.append(neural_network.predict("sigmoid"))
    