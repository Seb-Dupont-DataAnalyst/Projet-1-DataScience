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

df_final = pd.read_csv("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/result_file")

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

choice = st.sidebar.radio("", ('Accueil', 'Présentation Générale', "Analyse Exploratoire", 'Corrélation', "Tests d'hypothèse",'Prévisions de Prix', 'Fichier CSV'))

if choice == "Accueil":

    st.title("")
    st.title("")
    st.markdown('<body class="p">Bonjour et bienvenue pour cette présentation</body>', unsafe_allow_html=True)
    
    st.title("")
    col1, col2, col3 = st.columns([1,4,1])
    with col1:
      st.write("")
    with col2:
      st.image("https://miro.medium.com/max/1400/1*3zdEDGlAT5WEpOEbUj_luA.jpeg")
    with col3:
      st.write("")

if choice == 'Présentation Générale':
  

  st.title("")
  st.title("")
 
  st.markdown('<body class="p">Présentation Générale</body>', unsafe_allow_html=True)
  st.title("")
  st.title("")
  st.subheader("Où se situe la ville de Ames ?")
  st.title("")
  data = pd.DataFrame({'awesome cities': ['Ames'], 'lat': [
                        42.034534], 'lon': [-93.620369]})
  st.map(data)
  st.header("")
  st.write("Dataset sur les ventes des propriétés immobilières individuelles de la ville d'Ames dans l'Iowa entre 2006 et 2010.")
  st.write("2919 lignes et 80 variables.")
  st.write("Parmi les variables :")
  st.write("- 23 nominales : différents types de logements, de garages, d'environnements...")
  st.write("- 23 ordinales : évaluation de certains équipements")
  st.write("- 14 discrètes : nombre d'équipements (cuisines, chambres, salles de bains...)")
  st.write("- 20 continues : dimensions (surfaces)")
  st.write("Dataset original scindé en 2 datasets de taille égale, l'un d'entraînement, l'autre de test (sans les prix de vente).")
      

if choice == "Analyse Exploratoire":

    
    st.title("")
    st.title("")
    st.markdown('<body class="p">Analyse Exploratoire</body>', unsafe_allow_html=True)
    st.title("")

    st.write("")
    years = df['YearBuilt'].sort_values(ascending=True).unique().tolist()
    

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
    df_count.columns = ['MSZoning',  'YearBuilt', 'BldgType','Count']

    df_count_inf = df_filtered_inf.groupby(['MSZoning', 'YearBuilt', 'BldgType'])['SalePrice'].count()
    df_count_inf = df_count_inf.reset_index()
    df_count_inf.columns = ['MSZoning',  'YearBuilt', 'BldgType','Count']
    
    code = ("""
      A   Agriculture
      C   Commercial
      FV  Floating Village Residential
      I   Industrial
      RH  Residential High Density
      RL  Residential Low Density
      RP  Residential Low Density Park
      RM  Residential Medium Density""")
    
    

    col1, col2 = st.columns([1,3])

    with col1:
      st.code(code, language = 'python')
    with col2:
      fig10 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
      fig10.add_trace(go.Pie(labels=df_count_inf['MSZoning'], values=df_count_inf['Count'], name="Before " + str(choix_années)), 1, 1)
        
      fig10.add_trace(go.Pie(labels=df_count['MSZoning'], values=df_count['Count'], name="Since " + str(choix_années)),
                1, 2)
      fig10.update_layout(title="<b>Sale zoning classification</b>",
                        title_x=0.5, title_font_family="Verdana")
      fig10.update_layout(
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Before ' + str(choix_années), x=0.168, y=0.5, font_size=20, showarrow=False),
                    dict(text='Since ' + str(choix_années), x=0.828, y=0.5, font_size=20, showarrow=False)])
      fig10.update_traces(hole=.4)

      st.plotly_chart(fig10, use_container_width=True)



    code = ("""
        1Fam    Single-family Detached
        2FmCon  Two-family Conversion; originally built as one-family dwelling
        Duplx   Duplex
        TwnhsE  Townhouse End Unit
        TwnhsI  Townhouse Inside Unit""")
    
    

    col1, col2 = st.columns([1,3])

    with col1:
      st.code(code, language = 'python')
    with col2:
      fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
      fig.add_trace(go.Pie(labels=df_count_inf['BldgType'], values=df_count_inf['Count'], name="Before " + str(choix_années)), 1, 1)
        
      fig.add_trace(go.Pie(labels=df_count['BldgType'], values=df_count['Count'], name="Since " + str(choix_années)),
                1, 2)
      fig.update_layout(title="<b>Building Type Sale Classification</b>",
                        title_x=0.5, title_font_family="Verdana")
      fig.update_layout(
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Before ' + str(choix_années), x=0.17, y=0.5, font_size=20, showarrow=False),
                    dict(text='Since ' + str(choix_années), x=0.828, y=0.5, font_size=20, showarrow=False)])
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
    
    
    fig = px.box(df, x="GarageType", y="SalePrice", color = "GarageType")
    #fig.update_yaxes(range= [0, 600000])
    fig.update_layout(showlegend=False)
    fig.update_traces(quartilemethod="exclusive")
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_layout(title='<b>Prices by Garage Type</b>',
                       title_x=0.5, title_font_family="Verdana", showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

if choice == 'Corrélation':

    st.title("")
    st.title("")
    st.markdown('<body class="p">Analyse des corrélations</body>', unsafe_allow_html=True)
    st.title("")
    st.write("")

    pmatrix = df.corr().nlargest(9, columns="SalePrice")["SalePrice"].index

    coeffc=np.corrcoef(df[pmatrix].values.T)

    fig, axes = plt.subplots(figsize=(10, 5))
    sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
    st.write(fig)



if choice == "Tests d'hypothèse":
  
    sub_choice= st.sidebar.radio("Tests d'hypothèse", ('Test 1', 'Test 2')) 

    if sub_choice == 'Test 1' :
      st.title("")
      st.title("")
      st.markdown("<body class='p'>Tests d'hypothèse 1</body>", unsafe_allow_html=True)
      
      st.title("")
      st.write("")
      st.write('AFFIRMATION : "Le prix des maisons situées en zone résidentielle moyenne est plus élevé que celui des maisons situées en zones commerciales.')
      st.title("") 
      code = ("""
      A   Agriculture
      C   Commercial
      FV  Floating Village Residential
      I   Industrial
      RH  Residential High Density
      RL  Residential Low Density
      RP  Residential Low Density Park
      RM  Residential Medium Density""")
    
    

      col1, col2 = st.columns([1,3])

      with col1:
        st.code(code, language = 'python')
      with col2:
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
      st.write("Utilisation d'un t-test")
      #st.write("Paramètres utilisés : Moyennes des maisons situées en zones RM et en zones C")
      st.write("H0 : mean(RM) = mean(C)")
      st.write("H1 : mean(RM) ≠ mean(C)")
    
      RM = df[df['MSZoning'] == 'RM']['SalePrice']
      C = df[df['MSZoning'] == 'C (all)']['SalePrice']

      RM_mean = round(np.mean(RM),1)
      C_mean = round(np.mean(C),1)

      st.write("RM mean value:",RM_mean)
      st.write("C mean value:",C_mean)

      RM_std = round(RM.std(),1)
      C_std = round(C.std(),1)

      #st.write("RM std value:",RM_std)
      #st.write("C std value:",C_std)

      ttest,pval = ttest_ind(RM,C)

      st.write("p-value",round(pval,5))

      if pval <0.05:
        st.subheader("Rejet de l'hypothèse nulle : la moyenne des prix des maisons situées dans les zones RM est significativement différente de celle des maisons situées en zones C")
      else:
        st.subheader("We Accept the null hypothesis")
               
      ttest,pval = ttest_ind(RM,C, alternative ='greater')

      st.write("H0 : mean(RM) = mean(C)")
      st.write("H1 : mean(RM) > mean(C)")
      
      st.write("p-value",round(pval,5))

      if pval <0.05:
        st.subheader("Rejet de l'hypothèse nulle : la moyenne des prix des maisons situées dans les zones RM est significativement supérieure à celle des maisons situées en zones C")
      else:
        st.subheader("We Accept the null hypothesis")
        
    if sub_choice == 'Test 2' :
      
       st.title("")
       st.title("")

       st.markdown("<body class='p'>Tests d'hypothèse 2</body>", unsafe_allow_html=True)
       st.title("")
       st.write("")
       st.write('SUPPOSITION : "Le prix des maisons situées dans les villages flottants est plus élevé que celui des maisons situées en zone résidentielle."')
       code = ("""
       A   Agriculture
       C   Commercial
       FV  Floating Village Residential
       I   Industrial
       RH  Residential High Density
       RL  Residential Low Density
       RP  Residential Low Density Park
       RM  Residential Medium Density""")
    
    

       col1, col2 = st.columns([1,3])

       with col1:
         st.code(code, language = 'python')
       with col2:
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
        
       st.write("Utilisation d'un t-test")
       #st.write("Paramètres utilisés : Moyennes des maisons situées en zones RM et en zones C")
       st.write("H0 : mean(FV) = mean(R)")
       st.write("H1 : mean(FV) ≠ mean(R)")
    
       R = df[(df['MSZoning'] == 'RM') | (df['MSZoning'] == 'RL') | (df['MSZoning'] == 'RH')]['SalePrice']
       FV = df[df['MSZoning'] == 'FV']['SalePrice']

       R_mean = round(np.mean(R),1)
       FV_mean = round(np.mean(FV),1)

       st.write("R mean value:",R_mean)
       st.write("FV mean value:",FV_mean)

       R_std = round(R.std(),1)
       FV_std = round(FV.std(),1)

       #st.write("R std value:",R_std)
       #st.write("FV std value:",FV_std)
 
       ttest,pval = ttest_ind(FV,R)

       st.write("p-value",round(pval,5))

       if pval <0.05:
        st.subheader("Rejet de l'hypothèse nulle : la moyenne des prix des maisons situées dans les villages flottants est significativement différente de celle des maisons situées en zones R")
       else:
        st.subheader("We Accept the null hypothesis")
               
       ttest,pval = ttest_ind(FV,R, alternative ='greater')

       st.write("H0 : mean(FV) = mean(R)")
       st.write("H1 : mean(FV) > mean(R)")
      
       st.write("p-value",round(pval,5))

       if pval <0.05:
         st.subheader("Rejet de l'hypothèse nulle : la moyenne des prix des maisons situées dans les villages flottants est significativement supérieure à celle des maisons situées en zones R")
       else:
         st.subheader("We Accept the null hypothesis")


if choice == "Prévisions de Prix":
    
    sub_choice2= st.sidebar.radio("Prévisions de Prix", ('Choix du modèle de ML', 'Prévisions des Prix de Vente'))
    if sub_choice2 == 'Choix du modèle de ML': 
      st.title("")
      st.title("")
      st.markdown('<body class="p">Prévisions de Prix</body>', unsafe_allow_html=True)
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
      st.header("")
      st.subheader("Choix du modèle Gradient Boosting Regressor qui présente les meilleurs scores, pas d'overtfitting et la RMSE la plus faible")
      df_viz = pd.DataFrame()
      df_viz['real'] = y
      df_viz['predict'] = modelGBR.predict(X)
      df_viz = df_viz.sort_values(by='real')
      df_viz = df_viz.reset_index()
    
    
    if sub_choice2 == 'Prévisions des Prix de Vente': 
      st.title("")
      st.title("")
      st.markdown('<body class="p">Prévisions de Prix</body>', unsafe_allow_html=True)
      df_tot2 = transfo(df)
      y = df['SalePrice']
      X = df_tot2.iloc[:,0:-2]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
      modelGBR = GradientBoostingRegressor(subsample=1.0,n_estimators=68,min_samples_split=4,max_depth=4,learning_rate=0.06)
      modelGBR.fit(X_train, y_train)
      df_viz = pd.DataFrame()
      df_viz['real'] = y
      df_viz['predict'] = modelGBR.predict(X)
      df_viz = df_viz.sort_values(by='real')
      df_viz = df_viz.reset_index()  
      
      st.title("")
      st.title("")
      
      
      fig = px.scatter(df_viz, x=df_viz.index, y=["predict","real"], labels={"_index":"houses","value":"saleprice","variable":"Saleprice"})
      fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                      'paper_bgcolor': 'rgba(0,0,0,0)', })
      fig.update_layout(title="<b>Compararaison des prédictions avec les prix réels</b>",
                          title_x=0.5, title_font_family="Verdana")
      
      st.plotly_chart(fig, use_container_width=True)
      
if choice == 'Fichier CSV' : 
    st.title("")
    st.title("")
    st.markdown('<body class="p">Fichier CSV</body>', unsafe_allow_html=True)
    st.title("")
    st.title("")
    df_test9 = transfo(df_test)
    df_test9.dropna(axis=0, inplace=True)  
    df_test9['SalePrice_Pred'] = modelGBR.predict(df_test9[liste_col]).round(decimals=0)
    df_result = df_test9[['Id','SalePrice_Pred']]
    df_result = df_result.rename(columns={"SalePrice_Pred": "SalePrice"})
    df_result = df_result.set_index('Id')
    
    st.write(df_final.head(10))
    st.write('prix de vente moyen:',df_result.mean())
    st.write('écart type:',df_result.std())
      
