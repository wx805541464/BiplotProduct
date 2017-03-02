# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 15:35:52 2017

@author: wxun
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties
from sklearn import preprocessing
from xlwt import *
from xlutils.copy import copy
import xlsxwriter

workbook = xlsxwriter.Workbook('biplotdata.xlsx')
worksheet1 = workbook.add_worksheet('InputMatrix')
worksheet2 = workbook.add_worksheet('AdjustedMatrix')
worksheet3 = workbook.add_worksheet('Coordinates')
worksheet4 = workbook.add_worksheet('Variance')

def toexcel(headings,buname,ds,worksheet):
    data = pd.np.array(ds).tolist()
    #dataframe转二维list
    worksheet.write_row('A1', headings)   
    # 从$A$1位置开始横向把内容标题写入  
    worksheet.write_column('A2', buname)       
    # 从$A$2位置开始纵向把内容写入  
    for i in range(1,len(ds.index)+1):
        for j in range(1,len(ds.columns)+1):
            worksheet.write(i,j,data[i-1][j-1]) 
            
def pretext(): 
    my_csv = 'data/sheetfour.csv' # path to your dataset
    ds = pd.read_csv(my_csv)
    headings = list(ds.columns)  
    headings.insert(0,'')
    buname = list(ds.index)
    toexcel(headings,buname,ds,worksheet1)

    #对输入矩阵进行预处理
    meancol = ds.mean(axis=0)
    meanrow = ds.mean(axis=1)
    ds = ds.T
    ds = ds - meanrow
    ds = ds.T
    ds = ds - meancol
    ds = ds.T
    ds2 = ds.T
    toexcel(headings,buname,ds2,worksheet2)
    return ds

def PCAmethod(ds):
    #利用主成分分析进行降维
    X = pd.DataFrame(ds)
    n = len(X.columns)
    pca = PCA(n_components = n)
    X_pca = pca.fit(X).transform(X)    
    df_pca = pd.DataFrame(X_pca)
    #df_pca[0]和df_pca[1]表示产品的坐标值，df_pca是pca处理之后的降维数据
    
    df_pca.index = list(X.index)
    #修改df_pca的columns值
    sline = []
    for i in range(1,len(df_pca.columns)+1):  
        sline.append("%s" % i)
    df_pca.columns = sline
    return pca,df_pca,X,sline
    
def Biplot_plot(pca,df_pca,X,sline,ds):
    font = FontProperties(fname=r"c:\windows\fonts\msyh.ttf", size=10) 
    # Scatter plot based and assigne 
    sns.lmplot('1','2', data=df_pca, fit_reg = False, size = 10,scatter_kws={"s": 38,"color":"red"}).savefig('Biplot')   
    #pca.components_ 返回具有最大方差的成分
    #利用前两维度的数据组成属性在图中的坐标
    xvector = pca.components_[0]
    yvector = pca.components_[1]
   
    xs = pca.transform(X)[:,0]
    ys = pca.transform(X)[:,1]

    x_extends = max(abs(xs))*3
    y_extends = max(abs(ys))*2
    amplify_dim = []
    amplify_dim.insert(0,x_extends)
    amplify_dim.insert(1,y_extends)
    #每个维度扩大的倍数，即乘以该维最大的值
    for i in range(2,len(pca.transform(X))):
        amplify_dim.insert(i,max(abs(pca.transform(X)[:,i])))    
    pcacom = pd.DataFrame(pca.components_).T
    pcacom = pcacom*amplify_dim
    
    pcacom.index = list(X.columns)
    pcacom.columns = sline
    combine = [pcacom,df_pca]
    pcacoor = pd.concat(combine)
    
    headings = list(pcacoor.columns)  
    headings.insert(0,'')
    buname = list(pcacoor.index)
    toexcel(headings,buname,pcacoor,worksheet3)

    dfvar = pd.DataFrame(pca.explained_variance_ratio_[0:])
    pcaratio = []
    ratio = 0
    for i in range(len(pca.explained_variance_ratio_)):
        ratio += pca.explained_variance_ratio_[i]
        pcaratio.insert(i,ratio)
    dfratio = pd.DataFrame(pcaratio,columns=["Cumulative"])
    dfvar.columns = ["variance"]
    merge = [dfvar,dfratio]
    dfvar = pd.concat(merge,axis=1)
    columname = []
    for i in range(len(dfvar.index)):
        columname.insert(i,i+1)
    
    headings = list(dfvar.columns)  
    headings.insert(0,'')
    buname =  columname
    toexcel(headings,buname,dfvar,worksheet4)

    xvector_max = max(abs(xvector))
    yvector_max = max(abs(yvector))
    # value of the first two PCs, set the x, y axis boundary
    x_edge = max(abs(xs))*1.5
    y_edge = max(abs(ys))*1.5
    
    if xvector_max*x_extends > x_edge:
        x_edge = xvector_max*x_extends*1.3
        plt.xlim(x_edge,-x_edge)
    else:
        plt.xlim(x_edge,-x_edge) 
    if yvector_max*y_extends > y_edge:
        y_edge = yvector_max*y_extends*1.3
        plt.ylim(-y_edge,y_edge)
    else:
        plt.ylim(-y_edge,y_edge)
    
    # visualize projections
    # arrows project features (ie columns from csv) as vectors onto PC axes
    # we can adjust length and the size of the arrow
    #属性的坐标标识及绿色圆形，原点到属性的箭头
    for i in range(len(xvector)):
        plt.arrow(0, 0, xvector[i]*x_extends, yvector[i]*y_extends,color='r', width=0.005, head_width=0.05)
        plt.plot(xvector[i]*x_extends, yvector[i]*y_extends,'o',color='c')
        plt.text(xvector[i]*x_extends, yvector[i]*y_extends,list(ds.columns.values)[i], color='black',fontproperties=font)
    
    #产品的坐标标识
    for i in range(len(xs)):
        plt.text(xs[i]+1, ys[i], list(X.index)[i], color='b')
        
    plt.savefig('Biplot',dpi=300)    
    
 
if __name__ =="__main__":
    df = pretext()
    a,b,c,d = PCAmethod(df)
    Biplot_plot(a,b,c,d,df)
    workbook.close()