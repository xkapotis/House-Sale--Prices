import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as st
import missingno as msno
from scipy.stats import spearmanr
import sklearn.metrics as m
from scipy.spatial.distance import cdist




missing_values_set = ["","--","?","na","NAN","nan", '']         #For setting this values as nan
#sample_submission_data = pd.read_csv("./sample_submission.csv", na_values=missing_values_set)
test_data = pd.read_csv("./test.csv",na_values=missing_values_set)
train_data = pd.read_csv("./train.csv",na_values=missing_values_set)




#################### correlation train set (with SalePrice )

correlation_train = train_data.corr()
mask = np.zeros_like(correlation_train)                 # create a
mask[ np.triu_indices_from(mask)] = True                # triangle heatmap
sns.heatmap(correlation_train, mask=mask ,xticklabels=correlation_train.columns ,yticklabels=correlation_train.columns, linewidths=0.6 ,cmap='Blues')
plt.title("Train set Correlation")
plt.show()
plt.close()



#drop salePrice and concat train-test
frames = [ train_data, test_data]
full_dataset = pd.concat(frames,sort=False,ignore_index=True)       #merge train & test set and create  full_dataset dataframe
full_dataset=pd.DataFrame(data=full_dataset)
SalePrice = full_dataset["SalePrice"]
full_dataset.drop(columns=['SalePrice'])        # drop SalePrice from set
SalePrice.fillna("",inplace=True)               #After concat occurs Nan values from empty test[SalePrice] values---> fill wit ""
ID = pd.DataFrame( data = full_dataset["Id"])   #Get Id's






###data Quality


msno.matrix(full_dataset,labels=True)
plt.xticks(rotation='vertical',fontsize=10)
plt.title("Empty values in features")
plt.show()
plt.close()

# print(full_dataset.info())











###### Preprocessing
###### search for empty values in set
many_empty = [  attribute for attribute
                in full_dataset if
                full_dataset[attribute].isnull().sum() > 600     ] ##['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']


full_dataset = full_dataset.drop(columns=many_empty)    #### drop columns with over of 600 emty values

##### get numerical and categorical features
numerical_attr =[i for i in full_dataset.select_dtypes(include='number')]           # lenght :37
categorical_attr = [i for i in full_dataset.select_dtypes(include='object')]        # leght : 39







#### fill missing values  numerical ---> np.mean    , categorigal--->  most frequent value
for attributes in full_dataset:
    if attributes in numerical_attr:
        full_dataset[attributes] =full_dataset[attributes].fillna(value=full_dataset[attributes].median())
    else:
        full_dataset[attributes]=full_dataset[attributes].fillna(value=full_dataset[attributes].value_counts().idxmax() )






#### Convert categorical to numerical ( with dummies features ) and search for important ctegorical features with RFC & SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from  sklearn.model_selection import  train_test_split
#for avoiding overfittinG is a good practice to select the features by examining only the training set

X=pd.DataFrame(data=train_data[categorical_attr])           # create
# print(X)
X.drop(columns=['SalePrice'], inplace=True)                 # X , Y sets
Y=train_data['SalePrice']                                   # from train set
print("X , Y lenght of columns{} ,{} ".format(len(X.index)  , len(Y.index)))

for at in X:
    X = pd.concat([ X , pd.get_dummies(X[at] ,prefix=at , drop_first=True ) ],axis=1,sort = False )     # convert ctagorical features into dummies
    X.drop(columns=[at] , inplace=True)                                                                 # and then drop the feature

X_train_st , X_test_st , y_train_st, y_test_st  = train_test_split(X,Y, test_size=0.25)                    # split the X,Y sets

rfc = RandomForestClassifier(n_estimators=150)          #Create tree,  n_estimator = number of trees
sel = SelectFromModel(estimator=rfc,threshold='mean')                    # estimator = RFC
sel.fit(X,Y)                                            # fitting RFC on data
#print("threshold : , ",sel.estimator_.feature_importances_.mean())

#print(sel.get_support())                                               ##  Get a mask, or integer index, of the features selected
selected_features = X_train_st.columns[(sel.get_support())]             ## Get the features names
print("selected featurees lenght",len(selected_features))
print(selected_features)

importances = sel.estimator_.feature_importances_ #get feature importances
Features_importance = pd.DataFrame(data=X.columns.values , columns=['Features'])
Features_importance['Importance'] = importances
Features_importance=Features_importance.sort_values(by=['Importance'],axis=0,ascending=False)
print(Features_importance)

print("IMPORTANCES \n",importances)
print("len importances : \n",len(importances))
indices = np.argsort(importances)[::-1]                 #Returns the indices that should sort an array
print("len indices :", indices)
print("INDICES :\n ",len(indices))
for i in range(X.shape[1]):
    print("indices {}  :, {}".format(indices[i],importances[indices[i]]))
plt.figure(figsize=(10,10))                             ## plot
plt.title("Feature importances")                        ## features importances
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), Features_importance['Features'] , rotation='vertical')
plt.xlim([-1, 11.5])
plt.show()
plt.close()


#
# Index(['MSZoning_RL', 'MSZoning_RM', 'LotShape_Reg', 'LandContour_Lvl',               these features have importance > mean importance
#        'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside',
#        'Neighborhood_CollgCr', 'Neighborhood_Edwards'...
#
# DATAFRAME : Features_importance
#             Features  Importance
# 7        LotShape_Reg    0.025706
# 15   LotConfig_Inside    0.024541
# 171  GarageFinish_RFn    0.024006
# 132   BsmtExposure_No    0.023981
# 137  BsmtFinType1_Unf    0.021935

# SO:  CATEGORICAL TO NUMERICAL : "GarageFinish" ,"LotShape", "BsmtExposure", "BsmtFinType1","LotConfig"
#
cat_to_num = ["GarageFinish" ,"LotShape", "BsmtExposure", "BsmtFinType1","LotConfig"]

for categoricals in cat_to_num:                 # covnert the features [] into  dummies and drop the features
        full_dataset = pd.concat([ full_dataset , pd.get_dummies(full_dataset[categoricals] ,prefix=categoricals , drop_first=True ) ],axis=1,sort = False )
        full_dataset.drop(columns=[categoricals] , inplace=True)
        print("CATEGORICAL : ",categoricals)
        print(full_dataset)


numerical_attr =[i for i in full_dataset.select_dtypes(include='number')]



############## full dataset correlation
correlation = full_dataset.corr()
mask = np.zeros_like(correlation)                              # create a
mask[ np.triu_indices_from(mask)] = True                       # triangle correlation
sns.heatmap(correlation, mask=mask ,xticklabels=correlation.columns ,yticklabels=correlation.columns, linewidths=0.6 ,cmap='Blues')
plt.title('Correlation - final set')
plt.show()
plt.close()





################## katataksh A
correlation['CORR'] = correlation.sum(axis=0)                   # summarize each row and create a new column with name 'CORR' in correlation df
katataksh_A = correlation['CORR'].sort_values(ascending=False)  #sort values by ascending order of corelation
print("Katataksh A",katataksh_A)
#print(katataksh_A)  #first 5 : #GrLivArea   # OverallQua    # GarageCars      # GarageArea     FullBath
katataksh_A.to_csv(r"C:\Users\tzav2\Desktop\ΠΜΣ\python\submission\katataksh_A.csv",index=False,header=False)



#plot of LotArea
# plt.hist(full_dataset['LotArea'],bins=200)
# plt.xlim(-1,70000)
# plt.title("LotArea")
# plt.show()
# plt.close()


# ################# katataksh B
katataksh_B=[]
for attribute in numerical_attr:
    mean =  st.mean(full_dataset[attribute])
        # print('\n head of {} is : {} '.format(attribute,full_dataset[attribute].head()))
        # full_dataset[attribute].value_counts().plot(kind='hist',bins = 20)   #plot a hist of each feature
        # plt.title(attribute)
        # plt.show()
    stdv =  st.stdev( full_dataset[attribute] )
    var =   st.variance(full_dataset[attribute])
    katataksh_B.append([attribute,mean,var,stdv])

katataksh_B =pd.DataFrame(katataksh_B,columns=['ATTRIBUTE','Mean','Variance','Stdv']).sort_values(by=['Stdv'], ascending=False) #create the dataframe Katataksh_B
print("katataksh B \n",katataksh_B)
katataksh_B.to_csv(r"C:\Users\tzav2\Desktop\ΠΜΣ\python\submission\katataksh_B.csv",index=False)





########################## Standardization
##########################  z-score
from scipy.stats import zscore

old_full_dataset = full_dataset                                             #### get the old dataset before use Z-score
full_dataset[numerical_attr] = zscore(full_dataset[numerical_attr])        #    zscore in full_dataset
# print(full_dataset[numerical_attr])

######################## KATATAKSH C

z_corr  =full_dataset[numerical_attr].corr()
z_corr['zCORR'] = z_corr.sum(axis=0)
katataksh_C = z_corr['zCORR'].sort_values(ascending=False)
katataksh_C=  pd.DataFrame(katataksh_C)
print("katastash C: \n" ,katataksh_C)
katataksh_C.to_csv(r"C:\Users\tzav2\Desktop\ΠΜΣ\python\submission\katataksh_C.csv",index=False)





####################### KATATAKSH D


katataksh_D=[]
for attribute in numerical_attr:

    mean = st.mean(full_dataset[attribute])
    stdv = st.stdev(full_dataset[attribute])
    var = st.variance(full_dataset[attribute])
    katataksh_D.append([attribute, mean, var, stdv])

katataksh_D = pd.DataFrame(katataksh_D, columns=['ATTRIBUTE', 'Mean', 'Variance', 'Stdv']).sort_values(by=['Stdv'],
                                                                                                       ascending=False)
print("KATASTASH D \n" ,katataksh_D)


katataksh_D.to_csv(r"C:\Users\tzav2\Desktop\ΠΜΣ\python\submission\katataksh_D.csv",index=False)

############ pca for standardized data
from sklearn.decomposition import PCA
############ SEARCH FOR NUMBER OF COMPONENTS
pca_search = PCA().fit(full_dataset[numerical_attr])        #fit pca


# print(pca_search.explained_variance_ratio_)
fig, ax = plt.subplots()
print("PCA EXPLAINED VAR RATIO",pca_search.explained_variance_ratio_)
y = np.cumsum(pca_search.explained_variance_ratio_)          #get variance ratio cumsum for generate the plot  #cumsum[1,2,3]=[1,3,6] etc.
xi = np.arange(0, len(y), step=1)                                #number of features
plt.plot(xi,y,marker='o', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.72, color='r', linestyle='-')
plt.text(0.5, 0.72, '72% cut-off threshold', color = 'red', fontsize=12)
ax.grid(axis='x')
plt.show()
plt.close()


data_to_pca = full_dataset.drop(columns=["Id"])         #drop Id

numerical_attr_pca =[i for i in data_to_pca.select_dtypes(include='number')]        #get numeric attributes from full_dataset , after dropping Id




pca=PCA(n_components=20)                #n = 20 components
katataksi_e = pca.fit_transform(data_to_pca[numerical_attr_pca])
katataksi_e =pd.DataFrame(data=katataksi_e , columns=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20']) #create df :katataksh_e


katataksi_e.to_csv(r"C:\Users\tzav2\Desktop\ΠΜΣ\python\submission\katataksh_E.csv",index=False)



print("E (P20 COMPONENTS ): \n ",katataksi_e)

print("var ratio",pca.explained_variance_ratio_) # explain variance ratio of each Pi

print("Pca . components_")


#print("PCA . components_  leght :",len( pca.components_))

pca_components = pd.DataFrame(data=pca.components_ , columns=data_to_pca[numerical_attr_pca].columns)  # from -1 to 1 how much of each feature was used for create component Pi
print(pca_components.sum(axis=0))
print("components dataframe : \n",pca_components)








############ PCA in old (non-standardized) set
data_to_pca_OLD = old_full_dataset.drop(columns=["Id"])
numerical_attr_pca1 =[i for i in data_to_pca_OLD.select_dtypes(include='number')]
pca=PCA(n_components=20)
katataksi_F = pca.fit_transform(data_to_pca_OLD[numerical_attr_pca1])
katataksi_F =pd.DataFrame(data=katataksi_F , columns=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20'])
katataksi_F.to_csv(r"C:\Users\tzav2\Desktop\ΠΜΣ\python\submission\katataksh_F.csv",index=False)

print("SECOND PCA NON STANDARDIZED : \n")
print("var ratio of non standaerdized ",pca.explained_variance_ratio_) # explain variance ratio of each Pi
print("Pca components",pca.components_)
pca_components1 = pd.DataFrame(data=pca.components_ , columns=data_to_pca_OLD[numerical_attr_pca1].columns)  # from -1 to 1 how much of each feature was used for create component Pi
print(pca_components1)




#print("katataksh f : \n",katataksi_F)

pca_data = katataksi_e    #copy katataksh_e -> pca_data

from sklearn.cluster import KMeans

#how many clusters do we need?

inertias = []                     #inertia actually calculates the sum of all the points within a cluster from the centroid of that cluster.

distortions = []                  #Distortion: It is calculated as the average of the squared distances from the cluster centers of the respective clusters. Typically, the Euclidean distance metric is used.
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)



# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
plt.close()
#elbow with inertias :
# plt.plot(K, inertias, 'bx-')
# plt.xlabel('k')
# plt.ylabel('inertias')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()










clustering = KMeans(n_clusters=2, random_state=4 )          #n cluster :2
clustering.fit(pca_data)  #fit on data

predict = clustering.predict(pca_data)
pca_data["Cluster"] = predict                   #create columns
pca_data["Id"] =ID.values                       # Cluster(0 or 1) , Id , SalePrice
pca_data["SalePrice"] = SalePrice.values        # in pca_data

#create dataframes cluster_0 (all data in cluster 0 ) & cluster_1 ( all data in cluster 1 )
cluster_0 = pd.DataFrame(  data= pca_data[ pca_data['Cluster'] == 0 ] , columns=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','Id', 'Cluster','SalePrice']            )
cluster_1 = pd.DataFrame(  data= pca_data[ pca_data['Cluster'] == 1 ] , columns=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','Id', 'Cluster','SalePrice']            )


print("cluster 0 ",len(cluster_0))
print("cluster 1",len(cluster_1))
# print("CLUSTER 0 \n",cluster_0.info())
# print("CLUSTER 1 \n",cluster_1.info())
print('kmeans inertia :' ,kmeanModel.inertia_ ) # SUM OF DISTANCES FROM CENTROID



#### make train test and regression for each cluster :
results = []
id = []
clusters = [cluster_0 , cluster_1 ]

for i in range(0,2,1):          #create sets , train, test

        train = []
        test = []
        for row in clusters[i].values:

            x = row[22]         # SalePrice column
           # print(x)
            if x == "":                     #if is empty
                    test.append(row)        #append to test set
            else:
                   train.append(row)


        #create 2 dataframes train & test
        train_df = pd.DataFrame(train, columns=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','Id', 'Cluster','SalePrice'])
        test_df = pd.DataFrame(test, columns=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','Id', 'Cluster','SalePrice'])

        Id = pd.DataFrame(data=test_df["Id"])       #get Id of test set

        train_df.drop(columns=["Cluster"] , inplace=True)   #drop column
        test_df.drop(columns=["Cluster"] , inplace=True)    # 'Cluster'



        X_train = train_df.drop( columns=["Id","SalePrice"])                # Xtrain P1 to P20
        y_train = train_df["SalePrice"]                                     # Ytrain : SalePrice
                                                                            #
        X_test = test_df.drop(columns=["Id","SalePrice"])                   # Xtest : P1 to P20
        y_test =test_df["SalePrice"]                                        # Ytest : SalePrice
        print("len of empty in cluster {} is {} ".format(i,len(y_test)))

        from     sklearn.linear_model import LinearRegression

        regresion = LinearRegression(n_jobs=-1)                  #Create regression model
        regresion.fit(X_train,y_train)

        #Create polynominal for non linear regresion
        # from sklearn.preprocessing import PolynomialFeatures
        # from sklearn.linear_model import LassoCV
        # from sklearn.pipeline import make_pipeline
        #
        # lasso_eps = 0.0001
        # lasso_nalpha = 20
        # lasso_iter = 5000
        #
        # # Min and max degree of polynomials features to consider
        # degree = 2
        #
        # model = make_pipeline(PolynomialFeatures(degree, interaction_only=False),
        #                           LassoCV(eps=lasso_eps, n_alphas=lasso_nalpha, max_iter=lasso_iter,
        #                                   normalize=True, cv=5))
        # model.fit(X_train, y_train)

        prediction = regresion.predict(X_test)
        id.append(Id)                               #keep Id
        results.append(prediction)                  #keep results



results0 =pd.DataFrame(data=results[0] , columns=["cluster_pred"])      # results from first cluster
id0=id[0]                                                               #
results0["Id"] = id0.values

results1 =pd.DataFrame(data=results[1] , columns=["cluster_pred"])          #results from second cluster
id1 = id[1]                                                                 #
results1["Id"] =id1.values

#concat results
saleprice = pd.concat([results0,results1],axis=0 , sort=False)

saleprice.sort_values(by=['Id'] , inplace=True )                        #sort values by Id
saleprice.rename(columns={"cluster_pred":"SalePrice"} , inplace=True)   #rename Column cluster_pred to SalePrice

#create Sale Price Daataframe
saleprice = pd.DataFrame(data=saleprice , columns=["Id" , 'SalePrice'])


#export results to csv
# saleprice.to_csv(r"C:\Users\tzav2\Desktop\ΠΜΣ\python\submission\saleprice.csv" , index=False)

