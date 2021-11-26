import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsCV
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error
import math
from scipy.stats import ttest_ind
import streamlit as st

st.set_page_config(page_title="Project 1 - Analysis and predictions",
                   page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
                   
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)

df = pd.read_csv("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/train.csv")

df_test = pd.read_csv("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/test.csv")

st.markdown('<body class="title"> 📊 Project 1 - Analysis and predictions 📶</body>',
            unsafe_allow_html=True)

def transfo(df):

    dico = { 'Po' : 0, 'Fa' : 1, 'TA' : 2,  'Gd' : 3,'Ex' : 4}
    df['KitchenQual'] = df['KitchenQual'].map(dico)
    df['HeatingQC'] = df['HeatingQC'].map(dico)
    df['ExterQual'] = df['ExterQual'].map(dico)
    df['ExterCond'] = df['ExterCond'].map(dico)
    df2 = pd.get_dummies(df[['Neighborhood','GarageType']])
    result= pd.concat([df2,df[['OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual', 'GarageCars','GarageArea','TotalBsmtSF',
    '1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','Id']]], axis=1)

    return(result)

st.sidebar.title("Bonjour :bar_chart:")

choice = st.sidebar.selectbox("", ('Accueil', "Analyse Exploratoire", 'Corrélation', "Tests d'hypothèse",'Prévisions de Prix'))

if choice == "Accueil":
  
    st.title("")
  
    st.write("Bonjour, bienvenue pour cette présentation....")
    st.title("")
    
    st.image("https://miro.medium.com/max/1400/1*3zdEDGlAT5WEpOEbUj_luA.jpeg")

if choice == "Analyse Exploratoire":

    
    
    st.write("")
    st.write("")
    st.title("Analyse Exploratoire")
    st.title("")
    st.write("")
    years = df['YearBuilt'].sort_values(ascending=False).unique().tolist()
    

    #years.sort_values(ascending=False)



    choix_années = st.select_slider('Choisissez la période à étudier:',options=years)
    st.write("")

    df_filtered = df[df["YearBuilt"] >= choix_années]
    df_filtered_inf = df[df["YearBuilt"] < choix_années]

    fig = px.histogram(df_filtered, x="YearBuilt")
    fig.update_layout(title="<b>Houses by Year Built</b>",
    title_x=0.5, title_font_family="Verdana")
    fig.update_yaxes(title_text="<b>Houses")
    fig.update_xaxes(title_text="<b>Year Built</b>")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                   'paper_bgcolor': 'rgba(0,0,0,0)', })


    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df_filtered, x="SalePrice")
    fig.update_layout(title="<b>Houses by Price</b>",
    title_x=0.5, title_font_family="Verdana")
    fig.update_yaxes(title_text="<b>Houses")
    fig.update_xaxes(title_text="<b>Sale Price</b>")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                   'paper_bgcolor': 'rgba(0,0,0,0)', })


    st.plotly_chart(fig, use_container_width=True)

  
    fig = px.box(df_filtered, x="YearBuilt", y="SalePrice")
    fig.update_yaxes(range= [50000, 500000])
    fig.update_layout(title="<b>Sale Price by Year Built</b>",
        title_x=0.5, title_font_family="Verdana")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_yaxes(title_text="<b>Sale Price")
    fig.update_xaxes(title_text="<b>Year Built</b>")
    fig.update_traces(quartilemethod="exclusive")

    st.plotly_chart(fig, use_container_width=True)

    df_count = df_filtered.groupby(['MSZoning', 'YearBuilt', 'BldgType'])['SalePrice'].count()
    df_count = df_count.reset_index()
    df_count.columns = ['MSZoning',  'YearBuilt', 'BldgType', 'Count']

    df_count_inf = df_filtered_inf.groupby(['MSZoning', 'YearBuilt', 'BldgType'])['SalePrice'].count()
    df_count_inf = df_count_inf.reset_index()
    df_count_inf.columns = ['MSZoning',  'YearBuilt', 'BldgType', 'Count']

    fig10 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig10.add_trace(go.Pie(labels=df_count_inf['MSZoning'], values=df_count_inf['Count'], name="Before " + str(choix_années)), 1, 1)
        
    fig10.add_trace(go.Pie(labels=df_count['MSZoning'], values=df_count['Count'], name="Since " + str(choix_années)),
                1, 2)
    fig10.update_layout(title="<b>Sale zoning classification</b>",
                        title_x=0.5, title_font_family="Verdana")
    fig10.update_layout(
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Before ' + str(choix_années), x=0.176, y=0.5, font_size=20, showarrow=False),
                    dict(text='Since ' + str(choix_années), x=0.822, y=0.5, font_size=20, showarrow=False)])
    fig10.update_traces(hole=.4)

    st.plotly_chart(fig10, use_container_width=True)

    fig = px.box(df, x="MSZoning", y="SalePrice", color = "MSZoning")
    fig.update_yaxes(range= [0, 400000])
    fig.update_traces(quartilemethod="exclusive")
    fig.update_yaxes(title_text="<b>Sale Price")
    fig.update_xaxes(title_text="<b>MS Zoning</b>")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_layout(title='<b>Prices by MS Zoning</b>',
                       title_x=0.5, title_font_family="Verdana", showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=df_count_inf['BldgType'], values=df_count_inf['Count'], name="Before " + str(choix_années)), 1, 1)
        
    fig.add_trace(go.Pie(labels=df_count['BldgType'], values=df_count['Count'], name="Since " + str(choix_années)),
                1, 2)
    fig.update_layout(title="<b>Building Type Sale Classification</b>",
                        title_x=0.5, title_font_family="Verdana")
    fig.update_layout(
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Before ' + str(choix_années), x=0.178, y=0.5, font_size=20, showarrow=False),
                    dict(text='Since ' + str(choix_années), x=0.82, y=0.5, font_size=20, showarrow=False)])
    fig.update_traces(hole=.4)

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x="BldgType", y="SalePrice", color = "BldgType")
    fig.update_yaxes(range= [0, 400000])
    fig.update_traces(quartilemethod="exclusive")
    fig.update_yaxes(title_text="<b>Sale Price")
    fig.update_xaxes(title_text="<b>Building Type</b>")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_layout(title='<b>Prices by Building Type</b>',
                       title_x=0.5, title_font_family="Verdana", showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x="Neighborhood", y="SalePrice", color = "Neighborhood")
    fig.update_yaxes(range= [0, 600000])
    fig.update_layout(showlegend=False)
    fig.update_traces(quartilemethod="exclusive")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_layout(title='<b>Prices by Neighborhood</b>',
                       title_x=0.5, title_font_family="Verdana", showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

if choice == 'Corrélation':

    st.write("")
    st.write("")
    st.title("Analyse des corrélations")
    st.title("")
    st.write("")

    pmatrix = df.corr().nlargest(9, columns="SalePrice")["SalePrice"].index

    coeffc=np.corrcoef(df[pmatrix].values.T)

    fig, axes = plt.subplots(figsize=(10, 5))
    sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
    st.write(fig)



if choice == "Tests d'hypothèse":

    st.write("")
    st.write("")
    st.title("Tests d'hypothèse")
    st.title("")
    st.write("")

    code = ''' RM = df[df['MSZoning'] == 'RM']['SalePrice']
    C = df[df['MSZoning'] == 'C (all)']['SalePrice']

    RM_mean = np.mean(RM)
    C_mean = np.mean(C)

    RM_std = RM.std()
    C_std = C.std()

    ttest,pval = ttest_ind(RM,C)

    if pval <0.05:
        print("Reject the null hypothesis")
    else:
        print("Accept the null hypothesis")'''

    st.code(code, language='python')

    RM = df[df['MSZoning'] == 'RM']['SalePrice']
    C = df[df['MSZoning'] == 'C (all)']['SalePrice']

    RM_mean = round(np.mean(RM),1)
    C_mean = round(np.mean(C),1)

    st.write("RM mean value:",RM_mean)
    st.write("C mean value:",C_mean)

    RM_std = round(RM.std(),1)
    C_std = round(C.std(),1)

    st.write("RM std value:",RM_std)
    st.write("C std value:",C_std)

    ttest,pval = ttest_ind(RM,C)

    st.write("p-value",pval)

    if pval <0.05:
        st.subheader("We Reject the null hypothesis")
    else:
        st.subheader("We Accept the null hypothesis")


if choice == "Prévisions de Prix":

    st.write("")
    st.write("")
    st.title("Prévisions de Prix")
    st.title("")
    st.write("")

    df_tot2 = transfo(df)

    y = df['SalePrice']
    X = df_tot2.iloc[:,0:-2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    liste_col = X.columns.tolist()

    modelGBR = GradientBoostingRegressor(subsample=1.0,n_estimators=68,min_samples_split=4,max_depth=4,learning_rate=0.06)
    modelGBR.fit(X_train, y_train)
    score_train_GBR = modelGBR.score(X_train,y_train)
    score_test_GBR = modelGBR.score(X_test,y_test) 
    print('score_train=', score_train_GBR,'score_test=', score_test_GBR)

    modelRF = RandomForestRegressor(max_depth=7, min_samples_split=3, n_estimators=129, bootstrap=False, max_features='sqrt')
    modelRF.fit(X_train, y_train)
    score_train_RF = modelRF.score(X_train,y_train)
    score_test_RF = modelRF.score(X_test,y_test) 
    print('score_train=', score_train_RF,'score_test=', score_test_RF)

    modelKNN = KNeighborsRegressor(n_neighbors=6)
    modelKNN.fit(X_train, y_train)
    score_train_KNN = modelKNN.score(X_train,y_train)
    score_test_KNN = modelKNN.score(X_test,y_test) 
    print('score_train=', score_train_KNN,'score_test=', score_test_KNN)

    modelDTR = DecisionTreeRegressor(max_depth=3)
    modelDTR.fit(X_train, y_train)
    score_train_DTR = modelDTR.score(X_train,y_train)
    score_test_DTR = modelDTR.score(X_test,y_test) 
    print('score_train=', score_train_DTR,'score_test=', score_test_DTR)

    modelLR = LinearRegression()
    modelLR.fit(X_train, y_train)
    score_train_LR = modelLR.score(X_train,y_train)
    score_test_LR = modelLR.score(X_test,y_test) 
    print('score_train=', score_train_LR,'score_test=', score_test_LR)

    RMSE_LR = round(np.sqrt(mean_squared_log_error(y, modelLR.predict(X))),2)
    RMSE_KNN = round(np.sqrt(mean_squared_log_error(y, modelKNN.predict(X))),2)
    RMSE_DTR = round(np.sqrt(mean_squared_log_error(y, modelDTR.predict(X))),2)
    RMSE_RF = round(np.sqrt(mean_squared_log_error(y, modelRF.predict(X))),2)
    RMSE_GBR = round(np.sqrt(mean_squared_log_error(y, modelGBR.predict(X))),2)

    dict_RMSE = {'Gradient Boosting': RMSE_GBR, 'Random Forest' : RMSE_RF, 'Linear Regression' : RMSE_LR,
             'KNN' : RMSE_KNN, 'Decision Tree' : RMSE_DTR}

    values_RMSE = list(dict_RMSE.values())
    names_RMSE = list(dict_RMSE.keys())

    dict_score_test = {'Gradient Boosting': round(score_test_GBR,2), 'Random Forest' : round(score_test_RF,2), 'Linear Regression' : round(score_test_LR,2),
             'KNN' : round(score_test_KNN,2), 'Decision Tree' : round(score_test_DTR,2)}

    values_score_test = list(dict_score_test.values())
    names_score_test = list(dict_score_test.keys())

    dict_score_train = {'Gradient Boosting': round(score_train_GBR,2), 'Random Forest' : round(score_train_RF,2), 'Linear Regression' : round(score_train_LR,2),
         'KNN' : round(score_train_KNN,2), 'Decision Tree' : round(score_train_DTR,2)}

    values_score_train = list(dict_score_train.values())
    names_score_train = list(dict_score_train.keys())

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=names_RMSE, y=values_RMSE, text = values_RMSE,
                        mode='lines+markers+text',
                        name='RMSE'),secondary_y=False)
    fig.add_trace(go.Scatter(x=names_score_test, y=values_score_test, text = values_score_test,
                        mode='lines+markers+text',
                        name='Test Scores'), secondary_y=True)
    fig.add_trace(go.Scatter(x=names_score_train, y=values_score_train, text = values_score_train,
                        mode='lines+markers+text',
                        name='Train Scores'), secondary_y=True)
    fig.update_yaxes(title_text="<b>RMSE</b>", secondary_y=False)
    fig.update_traces(texttemplate='%{text:.1}',textposition='top center')
    fig.update_yaxes(title_text="<b>Scores</b>", secondary_y=True, range= [0, 1])
    fig.update_xaxes(title_text="<b>ML Models</b>")
    fig.update_layout(title="<b>ML Models comparison</b>",
                        title_x=0.5, title_font_family="Verdana")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)', })
    #fig.update_layout(width=900, height=500)
    

    st.plotly_chart(fig, use_container_width=True)
