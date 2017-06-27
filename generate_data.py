#coding:utf-8

from pyspark import SparkContext,SQLContext
from pyspark import Row
import numpy as np

##初始化
sc = SparkContext()
sqlContext = SQLContext(sc)

##读取数据
#data = sqlContext.read.json("data").select("imsi",'eci','btime','etime','new_serviceid','downoct','upoct')
data = sqlContext.read.json("/input/wangtong/updatefile_final").select("imsi",'eci','btime','etime','new_serviceid','downoct','upoct')

def splitData(line):
    words = line.split(",")
    return Row(eci = words[0],lat = float(words[1]),lon = float(words[2]))

##读取经纬度信息
#locationData = sc.textFile("transform.csv").map(splitData)
locationData = sc.textFile("/hhz/Snapshot/Data/transform.csv").map(splitData)

sqlContext.createDataFrame(locationData).registerTempTable("Temploc")
data.registerTempTable("TempData")

##获取 "用户在什么位置，打开了什么业务信息"--time granularity = 60mins
myDataFrame = sqlContext.sql('''select a.imsi,a.new_serviceid as serviceid,
 (a.downoct + a.upoct) as traffic,
 b.lon, b.lat,
 cast((unix_timestamp(a.btime)-unix_timestamp('2015-04-03 00:00:00'))/3600 as int) as mtime
 from TempData as a, Temploc as b where a.eci = b.eci''')
##sqlContext.sql返回的数据是dataFrame


####--------------------service and location preprocessing-------------------###
## pair RDD
def generateKV1(row):
    ServiceVec = np.zeros(15)    ##generate a vector to store a service
    LocVec = [[0,0]]
    ServiceVec[int(row.serviceid) - 1] = 1
    ServiceVec = ServiceVec.tolist()
    LocVec[0][0] = float(row.lon)
    LocVec[0][1] = float(row.lat)
    return ((row.imsi,row.mtime),(ServiceVec,LocVec))

##User aggregation
def userAgg1(x,y):
    AggService = np.array(x[0]) + np.array(y[0])
    AggService[AggService != 0] = 1
    AggService = AggService.tolist()
    AggLoc = x[1] + y[1]
    Aggloc_Distinct = []
    [Aggloc_Distinct.append(i) for i in AggLoc if i not in Aggloc_Distinct]
    return (AggService,Aggloc_Distinct)

def generateRow1(line):
    return Row(imsi = line[0][0], mtime = line[0][1], ServiceVec = line[1][0], LocVec = line[1][1])


FinalDataFrame = myDataFrame.rdd.map(generateKV1).reduceByKey(userAgg1).map(generateRow1)

#sqlContext.createDataFrame(FinalDataFrame).repartition(1).write.json("result4")


#####-------------------对每个用户，按mtime顺序聚合，生成ServiceMat 和 LocMat-----------------------##
def generateKV2(row):
    AggServiceVec = np.zeros((24,15))                   ##Service Mat
    AggServiceVec[row.mtime - 1] =row.ServiceVec
    AggServiceVec = AggServiceVec.tolist()
    AggLocVec = [[[0.0]]]*24                                  ##Loction list
    AggLocVec[row.mtime - 1] = row.LocVec
    return ((row.imsi),(AggServiceVec,AggLocVec))


def userAgg2(x,y):
    AggServiceMat = np.array(x[0]) + np.array(y[0])       ##Servicr Aggeration
    AggServiceMat = AggServiceMat.tolist()
    AggLocMat = [[[0.0]]]*24
    AggLocMat[x[1].index(max(x[1]))] = max(x[1])          ##Loction Aggregation
    AggLocMat[y[1].index(max(y[1]))] = max(y[1])
    return (AggServiceMat,AggLocMat)




def generateRow2(line):
    return Row(imsi = line[0], ServiceMat = line[1][0], LocMat = line[1][1])

AggDataFrame = FinalDataFrame.map(generateKV2).reduceByKey(userAgg2).map(generateRow2)

#sqlContext.createDataFrame(AggDataFrame).repartition(1).write.json("test_result1")

sqlContext.createDataFrame(AggDataFrame).write.json("/Johnson/User_behavior_analysis/data_10min")





