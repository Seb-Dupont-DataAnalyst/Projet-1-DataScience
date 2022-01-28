import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error, roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.cluster import Birch
from scipy.stats import ttest_ind, johnsonsu, norm, laplace, skewnorm, gennorm, chi2, tukeylambda, t, levy, chi
import streamlit as st


# CONFIG #

st.set_page_config(page_title="IOWA Project",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)

st.markdown("""
    <style>
    .titre {
        font-size:16px;
        font-weight:normal;
        margin:10px;
    }
    .text {
        font-size:25px;
        font-weight:normal;
        color:gray;
    }
    .sub_value {
        font-size:50px;
        font-weight:bold;
        line-height:1;
    }
    .value {
        font-size:80px;
        font-weight:bold;
        line-height:1;
    }
    </style>
    """, unsafe_allow_html=True)


# FONCTION #

def load_df(url):
    return pd.read_csv(url)


def transfo(df):
    """
    :param df: dataframe
    :return: dataframe with only numeric values
    """
    # Transformation des catégories en données numériques
    dico = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df['ExterQual'] = df['ExterQual'].map(dico)
    df['KitchenQual'] = df['KitchenQual'].map(dico)
    df2 = pd.get_dummies(df[['Neighborhood', 'GarageType']])

    data = pd.concat(
        [
            df2,
            df[
                [
                    'LotArea',
                    'LotFrontage',
                    'BsmtFinSF1',
                    'BsmtUnfSF',
                    'LotConfig',  # for LotFrontage regression
                    'LotShape',  # for LotFrontage regression
                    'OverallQual',
                    'GrLivArea',
                    'ExterQual',
                    'KitchenQual',
                    'GarageCars',
                    'GarageArea',
                    'TotalBsmtSF',
                    '1stFlrSF',
                    'FullBath',
                    'TotRmsAbvGrd',
                    'YearBuilt',
                    'YearRemodAdd',
                    'Id',
                    'SalePrice',
                    'BldgType',
                ]
            ],
        ],
        axis=1,
    )

    # Entrainement d'un model ML pour fill 'LotFrontage'
    cols_data = ['LotFrontage', 'LotArea', '1stFlrSF', 'LotConfig', 'LotShape']
    df = data[cols_data].copy()
    df['LotConfig'] = df['LotConfig'].factorize()[0]
    df['LotShape'] = df['LotShape'].factorize()[0]

    train = df[(data['LotFrontage'].notna()) & (data['LotArea'] < 30000)]
    predic = df[data['LotFrontage'].isna()]

    X = train[['LotArea', '1stFlrSF', 'LotConfig', 'LotShape']]
    y = train['LotFrontage']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, train_size=0.80)

    # parametres suite à un RandomizedSearch
    modelRFR = RandomForestRegressor(n_estimators=2000,
                                     min_samples_split=10,
                                     min_samples_leaf=1,
                                     max_features='sqrt',
                                     max_depth=100,
                                     bootstrap=True).fit(X_train, y_train)

    nan_cols = ['LotArea', '1stFlrSF', 'LotConfig', 'LotShape']
    temp = data[data['LotFrontage'].isna()]
    temp['LotFrontage'] = modelRFR.predict(predic[nan_cols])

    return pd.concat([data[data['LotFrontage'].notna()], temp]).drop(columns=['LotConfig', 'LotShape'])


def discret_layout(fig):
    """
    :param fig: a plotly fig with discret distribution
    :return: a fig with the good layout
    """
    fig.update_layout(font_family='IBM Plex Sans', uniformtext_minsize=10, uniformtext_mode='hide',
                      xaxis=dict(title=None),
                      margin=dict(l=10, r=10, b=10, t=10))
    return fig


def space(n):
    """
    :param n: number of spaces
    :return: print n spaces
    """
    for n in range(n):
        st.title(" ")


def value(number):
    """
    :param number: int or float
    :return: print the number in big
    """
    st.markdown(f'''<p class="sub_value">{number}</p>
                    ''', unsafe_allow_html=True)


# DATA #

df = load_df("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/train.csv")
df_test = load_df("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/test.csv")
df_final = load_df("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/result_file_1")
data = transfo(df)

localisation = pd.DataFrame({'awesome cities': ['Ames'], 'lat': [42.034534], 'lon': [-93.620369]})

# CALCULE DES DISTRIB (pour réduire le chargement entre les pages)

# valeurs obtenues après application d'un FITTER sur les données
grliv_dist = chi2.rvs(df=12.309473472853526,
                      loc=254.64499056971584, scale=102.4266951654377, size=10000)
totbsmt_dist = johnsonsu.rvs(a=-0.5917807657522386, b=1.3919373074176713,
                             loc=821.4459540582604, scale=423.0770582601845, size=10000)
ggeaera_dist = laplace.rvs(loc=479.9999999992406, scale=159.93355122237585, size=10000)
florsf_dist = skewnorm.rvs(a=4.514262333482848,
                           loc=698.8500285155108, scale=603.6857562658566, size=10000)
totrms_dist = gennorm.rvs(beta=0.2962186633639843,
                          loc=6.0, scale=0.007513870476739083, size=10000)
lotarea_dist = tukeylambda.rvs(lam=-0.3886757898461206,
                               loc=9414.12399652672, scale=1281.190016263276, size=10000)
lotfront_dist = t.rvs(df=3.7001417217447754,
                      loc=69.48591606554618, scale=16.53208231942333, size=10000)
bsmtfin_dist = levy.rvs(loc=-2.649602255093832, scale=8.107296031226449, size=10000)
bsmtnofin_dist = chi.rvs(df=0.7986648461917891,
                         loc=-5.807409708971971e-29, scale=804.9431885685528, size=10000)

fig_grliv = ff.create_distplot([data['GrLivArea'], grliv_dist[grliv_dist > 0]],
                               ['GrLivArea', 'Chi2 distribution'],
                               show_hist=False)
# fig_totbsmt = ff.create_distplot([data['TotalBsmtSF'], totbsmt_dist[totbsmt_dist > 0]],
# ['TotalBsmtSF', 'Johnsonsu distribution'],
# show_hist=False)
fig_ggeaera = ff.create_distplot([data['GarageArea'], ggeaera_dist],
                                 ['GarageArea', 'Laplace distribution'],
                                 show_hist=False)
fig_florsf = ff.create_distplot([data['1stFlrSF'], florsf_dist],
                                ['1stFlrSF', 'Skewnorm distribution'],
                                show_hist=False)
# fig_bsmtfin = ff.create_distplot([data['BsmtFinSF1'], bsmtfin_dist[bsmtfin_dist < 1000]],
# ['BsmtFinSF1', 'levy distribution'],
# show_hist=False)
# fig_bsmtnofin = ff.create_distplot([data['BsmtUnfSF'], bsmtnofin_dist],
# ['BsmtUnfSF', 'chi distribution'],
# show_hist=False)
fig_lotfront = ff.create_distplot([data['LotFrontage'], lotfront_dist[lotfront_dist > 0]],
                                  ['LotFrontage', 't distribution'],
                                  show_hist=False)
fig_lotarea = ff.create_distplot([data['LotArea'], lotarea_dist[lotarea_dist > 0]],
                                 ['LotArea', 'tukeylambda distribution'],
                                 show_hist=False)
for fig in [fig_grliv, fig_ggeaera, fig_florsf, fig_lotfront, fig_lotarea]:
    fig.update_layout(font_family='IBM Plex Sans',
                      yaxis=dict(visible=False),
                      uniformtext_minsize=10, uniformtext_mode='hide',
                      margin=dict(l=10, r=10, b=10, t=10),
                      legend=dict(x=1, y=1.02,
                                  orientation="h",
                                  yanchor="bottom",
                                  xanchor="right",
                                  bgcolor='rgba(0,0,0,0)',
                                  font=dict(size=12)))

# SIDEBAR #

st.sidebar.title("Table des matières")
week = st.sidebar.radio('Choix de la semaine :', ('Accueil', 'Semaine 1', 'Semaine 2', 'Semaine 3'))

# MAIN #

st.markdown('''<body class="title"> AMES' 🏠 PROJECT</body>''',
            unsafe_allow_html=True)

if week == "Accueil":
    space(1)
    st.markdown("<body class='p'> La Situation de l'Immobilier à AMES</body>", unsafe_allow_html=True)
    space(1)

    st.subheader("Où se situe la ville de Ames ?")
    st.map(localisation)

    space(1)
    st.markdown("La base de données ressence les ventes des propriétés immobilières individuelles de la ville d'Ames "
                "dans l'Iowa, entre **2006** et **2010**. Une moitié des données ont un prix de vente associé, "
                "alors que pour une autre moitié, le prix devra être prédit.")
    space(1)

    cols = st.columns(3)

    with cols[0]:
        st.markdown("La base de données comprend les éléments suivants : ")

    with cols[1]:
        st.markdown(f'''<p class="value">2919</p>
                        <p class="titre">Maisons répertoriées</p>
                     ''', unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f'''<p class="value">80</p>
                        <p class="titre">Caractéristiques différentes</p>
                     ''', unsafe_allow_html=True)

    space(1)
    cols = st.columns(4)
    with cols[0]:
        value(23)
        st.markdown("**nominales** (types de logements, de garages, etc)")
    with cols[1]:
        value(23)
        st.markdown("**ordinales** (évaluation de certains équipements)")
    with cols[2]:
        value(14)
        st.markdown("**discrètes** (nombre de cuisines, chambres, salles de bains, etc)")
    with cols[3]:
        value(20)
        st.markdown("**continues** (surfaces)")

if week == 'Semaine 1':
    choice = st.sidebar.radio("Semaine 1", ("Analyse Exploratoire", "Corrélation", "Tests d'hypothèse",
                                            "Prévisions de Prix", "Fichier CSV", "Conclusion"))

    if choice == "Analyse Exploratoire":
        space(1)
        st.markdown('<body class="p">Analyse Exploratoire</body>', unsafe_allow_html=True)

        years = df['YearBuilt'].sort_values(ascending=True).unique().tolist()
        space(1)
        choice_year = st.select_slider('Choisissez la période à étudier:', options=years)
        df_filtered = df[df["YearBuilt"] >= choice_year]
        df_filtered_inf = df[df["YearBuilt"] < choice_year]

        space(1)
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
        fig.update_yaxes(range=[50000, 500000])
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
        df_count.columns = ['MSZoning', 'YearBuilt', 'BldgType', 'Count']

        df_count_inf = df_filtered_inf.groupby(['MSZoning', 'YearBuilt', 'BldgType'])['SalePrice'].count()
        df_count_inf = df_count_inf.reset_index()
        df_count_inf.columns = ['MSZoning', 'YearBuilt', 'BldgType', 'Count']

        code = ("""
          A   Agriculture
          C   Commercial
          FV  Floating Village Residential
          I   Industrial
          RH  Residential High Density
          RL  Residential Low Density
          RP  Residential Low Density Park
          RM  Residential Medium Density""")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.code(code, language='python')
        with col2:
            fig10 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig10.add_trace(go.Pie(labels=df_count_inf['MSZoning'], values=df_count_inf['Count'],
                                   name="Before " + str(choice_year)), 1, 1)

            fig10.add_trace(
                go.Pie(labels=df_count['MSZoning'], values=df_count['Count'], name="Since " + str(choice_year)),
                1, 2)
            fig10.update_layout(title="<b>Sale zoning classification</b>",
                                title_x=0.5, title_font_family="Verdana")
            fig10.update_layout(
                # Add annotations in the center of the donut pies.
                annotations=[dict(text='Before ' + str(choice_year), x=0.168, y=0.5, font_size=20, showarrow=False),
                             dict(text='Since ' + str(choice_year), x=0.828, y=0.5, font_size=20, showarrow=False)])
            fig10.update_traces(hole=.4)

            st.plotly_chart(fig10, use_container_width=True)

        code = ("""
            1Fam    Single-family Detached
            2FmCon  Two-family Conversion; originally built as one-family dwelling
            Duplx   Duplex
            TwnhsE  Townhouse End Unit
            TwnhsI  Townhouse Inside Unit""")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.code(code, language='python')
        with col2:
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig.add_trace(go.Pie(labels=df_count_inf['BldgType'], values=df_count_inf['Count'],
                                 name="Before " + str(choice_year)), 1, 1)

            fig.add_trace(
                go.Pie(labels=df_count['BldgType'], values=df_count['Count'], name="Since " + str(choice_year)),
                1, 2)
            fig.update_layout(title="<b>Building Type Sale Classification</b>",
                              title_x=0.5, title_font_family="Verdana")
            fig.update_layout(
                # Add annotations in the center of the donut pies.
                annotations=[dict(text='Before ' + str(choice_year), x=0.17, y=0.5, font_size=20, showarrow=False),
                             dict(text='Since ' + str(choice_year), x=0.828, y=0.5, font_size=20, showarrow=False)])
            fig.update_traces(hole=.4)

            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x="BldgType", y="SalePrice", color="BldgType")
        fig.update_yaxes(range=[0, 400000])
        fig.update_traces(quartilemethod="exclusive")
        fig.update_yaxes(title_text="<b>Sale Price")
        fig.update_xaxes(title_text="<b>Building Type</b>")
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                           'paper_bgcolor': 'rgba(0,0,0,0)', })
        fig.update_layout(title='<b>Prices by Building Type</b>',
                          title_x=0.5, title_font_family="Verdana", showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x="Neighborhood", y="SalePrice", color="Neighborhood")
        fig.update_yaxes(range=[0, 600000])
        fig.update_layout(showlegend=False)
        fig.update_traces(quartilemethod="exclusive")
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                           'paper_bgcolor': 'rgba(0,0,0,0)', })
        fig.update_layout(title='<b>Prices by Neighborhood</b>',
                          title_x=0.5, title_font_family="Verdana", showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        df_garage = df.dropna(subset=['GarageType'])
        fig = px.box(df_garage, x="GarageType", y="SalePrice", color="GarageType")
        # fig.update_yaxes(range= [0, 600000])
        fig.update_layout(showlegend=False)
        fig.update_traces(quartilemethod="exclusive")
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                           'paper_bgcolor': 'rgba(0,0,0,0)', })
        fig.update_layout(title='<b>Prices by Garage Type</b>',
                          title_x=0.5, title_font_family="Verdana", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if choice == 'Corrélation':
        space(1)
        st.markdown('<body class="p">Analyse des corrélations</body>', unsafe_allow_html=True)
        st.title("")
        st.write("")

        pmatrix = df.corr().nlargest(9, columns="SalePrice")["SalePrice"].index

        coeffc = np.corrcoef(df[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(10, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1,
                    cmap="YlGnBu")
        st.write(fig)

    if choice == "Tests d'hypothèse":
        sub_choice = st.sidebar.radio("Tests d'hypothèse", ('Test 1', 'Test 2'))

        if sub_choice == 'Test 1':
            space(1)
            st.markdown("<body class='p'>Tests d'hypothèse 1</body>", unsafe_allow_html=True)

            st.title("")
            st.write("")
            st.subheader(
                'AFFIRMATION : "Le prix des maisons situées en zone résidentielle moyenne est plus élevé que celui des maisons situées en zones commerciales.')
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

            col1, col2 = st.columns([1, 3])
            with col1:
                st.code(code, language='python')
            with col2:
                fig = px.box(df, x="MSZoning", y="SalePrice", color="MSZoning")
                fig.update_yaxes(range=[0, 400000])
                fig.update_traces(quartilemethod="exclusive")
                fig.update_yaxes(title_text="<b>Sale Price")
                fig.update_xaxes(title_text="<b>MS Zoning</b>")
                fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                                   'paper_bgcolor': 'rgba(0,0,0,0)', })
                fig.update_layout(title='<b>Prices by MS Zoning</b>',
                                  title_x=0.5, title_font_family="Verdana", showlegend=False)

                st.plotly_chart(fig, use_container_width=True)
            st.write("Utilisation d'un t-test")
            # st.write("Paramètres utilisés : Moyennes des maisons situées en zones RM et en zones C")
            st.write("H0 : mean(RM) = mean(C)")
            st.write("H1 : mean(RM) ≠ mean(C)")

            RM = df[df['MSZoning'] == 'RM']['SalePrice']
            C = df[df['MSZoning'] == 'C (all)']['SalePrice']

            RM_mean = round(np.mean(RM), 1)
            C_mean = round(np.mean(C), 1)

            st.write("RM mean value:", RM_mean)
            st.write("C mean value:", C_mean)

            RM_std = round(RM.std(), 1)
            C_std = round(C.std(), 1)

            # st.write("RM std value:",RM_std)
            # st.write("C std value:",C_std)

            ttest, pval = ttest_ind(RM, C)

            st.write("p-value", round(pval, 5))

            if pval < 0.05:
                st.subheader(
                    "Rejet de l'hypothèse nulle : la moyenne des prix des maisons situées dans les zones RM est significativement différente de celle des maisons situées en zones C")
            else:
                st.subheader("We Accept the null hypothesis")

            ttest, pval = ttest_ind(RM, C, alternative='greater')

            st.write("H0 : mean(RM) = mean(C)")
            st.write("H1 : mean(RM) > mean(C)")

            st.write("p-value", round(pval, 5))

            if pval < 0.05:
                st.subheader(
                    "Rejet de l'hypothèse nulle : la moyenne des prix des maisons situées dans les zones RM est significativement supérieure à celle des maisons situées en zones C")
            else:
                st.subheader("We Accept the null hypothesis")

        if sub_choice == 'Test 2':
            space(1)
            st.markdown("<body class='p'>Tests d'hypothèse 2</body>", unsafe_allow_html=True)
            st.title("")
            st.write("")
            st.subheader(
                'SUPPOSITION : "Le prix des maisons situées dans les villages flottants est plus élevé que celui des maisons situées en zone résidentielle."')
            code = ("""
           A   Agriculture
           C   Commercial
           FV  Floating Village Residential
           I   Industrial
           RH  Residential High Density
           RL  Residential Low Density
           RP  Residential Low Density Park
           RM  Residential Medium Density""")

            col1, col2 = st.columns([1, 3])

            with col1:
                st.code(code, language='python')
            with col2:
                fig = px.box(df, x="MSZoning", y="SalePrice", color="MSZoning")
                fig.update_yaxes(range=[0, 400000])
                fig.update_traces(quartilemethod="exclusive")
                fig.update_yaxes(title_text="<b>Sale Price")
                fig.update_xaxes(title_text="<b>MS Zoning</b>")
                fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                                   'paper_bgcolor': 'rgba(0,0,0,0)', })
                fig.update_layout(title='<b>Prices by MS Zoning</b>',
                                  title_x=0.5, title_font_family="Verdana", showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

            st.write("Utilisation d'un t-test")
            # st.write("Paramètres utilisés : Moyennes des maisons situées en zones RM et en zones C")
            st.write("H0 : mean(FV) = mean(R)")
            st.write("H1 : mean(FV) > mean(R)")

            R = df[(df['MSZoning'] == 'RM') | (df['MSZoning'] == 'RL') | (df['MSZoning'] == 'RH')]['SalePrice']
            FV = df[df['MSZoning'] == 'FV']['SalePrice']

            R_mean = round(np.mean(R), 1)
            FV_mean = round(np.mean(FV), 1)

            st.write("R mean value:", R_mean)
            st.write("FV mean value:", FV_mean)

            R_std = round(R.std(), 1)
            FV_std = round(FV.std(), 1)

            ttest, pval = ttest_ind(FV, R, alternative='greater')

            st.write("p-value", round(pval, 5))

            if pval < 0.05:
                st.subheader(
                    "Rejet de l'hypothèse nulle : la moyenne des prix des maisons situées dans les villages flottants est significativement supérieure à celle des maisons situées en zone résidentielle")
            else:
                st.subheader("We Accept the null hypothesis")

    if choice == "Prévisions de Prix":
        sub_choice2 = st.sidebar.radio("Prévisions de Prix", ('Choix du modèle de ML', 'Prévisions des Prix de Vente'))

        if sub_choice2 == 'Choix du modèle de ML':
            space(1)
            st.markdown('<body class="p">Prévisions de Prix</body>', unsafe_allow_html=True)
            st.title("")
            st.write("")

            df_tot2 = transfo(df)

            y = df['SalePrice']
            X = df_tot2.iloc[:, 0:-2]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            liste_col = X.columns.tolist()

            modelGBR = GradientBoostingRegressor(subsample=1.0, n_estimators=68, min_samples_split=4, max_depth=4,
                                                 learning_rate=0.06)
            modelGBR.fit(X_train, y_train)
            score_train_GBR = modelGBR.score(X_train, y_train)
            score_test_GBR = modelGBR.score(X_test, y_test)
            print('score_train=', score_train_GBR, 'score_test=', score_test_GBR)

            modelRF = RandomForestRegressor(max_depth=7, min_samples_split=3, n_estimators=129, bootstrap=False,
                                            max_features='sqrt')
            modelRF.fit(X_train, y_train)
            score_train_RF = modelRF.score(X_train, y_train)
            score_test_RF = modelRF.score(X_test, y_test)
            print('score_train=', score_train_RF, 'score_test=', score_test_RF)

            modelKNN = KNeighborsRegressor(n_neighbors=6)
            modelKNN.fit(X_train, y_train)
            score_train_KNN = modelKNN.score(X_train, y_train)
            score_test_KNN = modelKNN.score(X_test, y_test)
            print('score_train=', score_train_KNN, 'score_test=', score_test_KNN)

            modelDTR = DecisionTreeRegressor(max_depth=3)
            modelDTR.fit(X_train, y_train)
            score_train_DTR = modelDTR.score(X_train, y_train)
            score_test_DTR = modelDTR.score(X_test, y_test)
            print('score_train=', score_train_DTR, 'score_test=', score_test_DTR)

            modelLR = LinearRegression()
            modelLR.fit(X_train, y_train)
            score_train_LR = modelLR.score(X_train, y_train)
            score_test_LR = modelLR.score(X_test, y_test)
            print('score_train=', score_train_LR, 'score_test=', score_test_LR)

            RMSE_LR = round(np.sqrt(mean_squared_log_error(y, modelLR.predict(X))), 2)
            RMSE_KNN = round(np.sqrt(mean_squared_log_error(y, modelKNN.predict(X))), 2)
            RMSE_DTR = round(np.sqrt(mean_squared_log_error(y, modelDTR.predict(X))), 2)
            RMSE_RF = round(np.sqrt(mean_squared_log_error(y, modelRF.predict(X))), 2)
            RMSE_GBR = round(np.sqrt(mean_squared_log_error(y, modelGBR.predict(X))), 2)

            dict_RMSE = {'Gradient Boosting': RMSE_GBR, 'Random Forest': RMSE_RF, 'Linear Regression': RMSE_LR,
                         'KNN': RMSE_KNN, 'Decision Tree': RMSE_DTR}

            values_RMSE = list(dict_RMSE.values())
            names_RMSE = list(dict_RMSE.keys())

            dict_score_test = {'Gradient Boosting': round(score_test_GBR, 2), 'Random Forest': round(score_test_RF, 2),
                               'Linear Regression': round(score_test_LR, 2),
                               'KNN': round(score_test_KNN, 2), 'Decision Tree': round(score_test_DTR, 2)}

            values_score_test = list(dict_score_test.values())
            names_score_test = list(dict_score_test.keys())

            dict_score_train = {'Gradient Boosting': round(score_train_GBR, 2),
                                'Random Forest': round(score_train_RF, 2),
                                'Linear Regression': round(score_train_LR, 2),
                                'KNN': round(score_train_KNN, 2), 'Decision Tree': round(score_train_DTR, 2)}

            values_score_train = list(dict_score_train.values())
            names_score_train = list(dict_score_train.keys())

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=names_RMSE, y=values_RMSE, text=values_RMSE,
                                     mode='lines+markers+text',
                                     name='RMSE'), secondary_y=False)
            fig.add_trace(go.Scatter(x=names_score_test, y=values_score_test, text=values_score_test,
                                     mode='lines+markers+text',
                                     name='Test Scores'), secondary_y=True)
            fig.add_trace(go.Scatter(x=names_score_train, y=values_score_train, text=values_score_train,
                                     mode='lines+markers+text',
                                     name='Train Scores'), secondary_y=True)
            fig.update_yaxes(title_text="<b>RMSLE</b>", secondary_y=False)
            fig.update_traces(texttemplate='%{text:.1}', textposition='top center')
            fig.update_yaxes(title_text="<b>Scores</b>", secondary_y=True, range=[0, 1])
            fig.update_xaxes(title_text="<b>ML Models</b>")
            fig.update_layout(title="<b>ML Models comparison</b>",
                              title_x=0.5, title_font_family="Verdana")
            fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                               'paper_bgcolor': 'rgba(0,0,0,0)', })
            # fig.update_layout(width=900, height=500)

            st.plotly_chart(fig, use_container_width=True)
            st.header("")
            st.subheader(
                "Choix du modèle Gradient Boosting Regressor qui présente les meilleurs scores, pas d'overfitting et la RMSLE la plus faible")
            df_viz = pd.DataFrame()
            df_viz['real'] = y
            df_viz['predict'] = modelGBR.predict(X)
            df_viz = df_viz.sort_values(by='real')
            df_viz = df_viz.reset_index()

        if sub_choice2 == 'Prévisions des Prix de Vente':
            space(1)
            st.markdown('<body class="p">Prévisions de Prix</body>', unsafe_allow_html=True)

            df_tot2 = transfo(df)
            y = df['SalePrice']
            X = df_tot2.iloc[:, 0:-2]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            modelGBR = GradientBoostingRegressor(subsample=1.0, n_estimators=68, min_samples_split=4, max_depth=4,
                                                 learning_rate=0.06)
            modelGBR.fit(X_train, y_train)
            df_viz = pd.DataFrame()
            df_viz['real'] = y
            df_viz['predict'] = modelGBR.predict(X)
            df_viz = df_viz.sort_values(by='real')
            df_viz = df_viz.reset_index()

            st.title("")
            st.title("")

            fig = px.scatter(df_viz, x=df_viz.index, y=["predict", "real"],
                             labels={"_index": "houses", "value": "saleprice", "variable": "Saleprice"})
            fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                               'paper_bgcolor': 'rgba(0,0,0,0)', })
            fig.update_layout(title="<b>Compararaison des prédictions avec les prix réels</b>",
                              title_x=0.5, title_font_family="Verdana")

            st.plotly_chart(fig, use_container_width=True)

    if choice == 'Fichier CSV':
        space(1)
        st.markdown('<body class="p">Fichier CSV</body>', unsafe_allow_html=True)

        space(2)
        cols = st.columns(3)
        with cols[0]:
            st.dataframe(df_final.head(10))
        with cols[1]:
            space(2)
            value(round(df_final['SalePrice'].mean()))
            st.markdown("Prix de vente moyen")
        with cols[2]:
            space(2)
            value(round(round(df_final['SalePrice'].std())))
            st.markdown("Ecart type")

    if choice == "Conclusion":
        space(2)

        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image("https://media.makeameme.org/created/merci-de-votre-5bd62e.jpg", width=800)

if week == 'Semaine 2':
    choice = st.sidebar.radio("Semaine 2", ("Previously", "Distributions", "Machine Learning", "Clustering"))

    if choice == "Previously":
        space(1)
        st.markdown('''<body class="p">Previoulsy on the AMES'PROJECT !</body>''', unsafe_allow_html=True)

        space(2)
        st.title("La base de travail")

        cols = st.columns(3)
        with cols[0]:
            st.subheader("Le nettoyage des données")
            st.image('https://static.thenounproject.com/png/2301589-200.png',
                     width=200)
        with cols[1]:
            st.subheader("L'étude des corrélations")
            space(1)
            st.write(' ')
            st.image(
                'https://www.pngkit.com/png/full/231-2316802_full-database-search-comments-database-search-icon-free.png',
                width=130)
        with cols[2]:
            st.subheader("Le choix des variables")
            st.header(" ")
            st.image(
                'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTuaDsQDYiaYPoJL2MmM9I5e1JnvLJH9M9SEglYfjkr6f5_RJnGQcDj1DGITrOILO-ssfk&usqp=CAU',
                width=170)

        space(2)
        st.title("Table des corrélations")

        pmatrix = df.corr().nlargest(13, columns="SalePrice")["SalePrice"].index
        coeffc = np.corrcoef(df[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(10, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values,
                    xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
        st.write(fig)

        space(2)
        st.title("Le poids des variables dans le Machine Learning")

        features = pd.DataFrame([['LotArea', 260],
                                 ['GrLivArea', 205],
                                 ['TotalBsmtSf', 180],
                                 ['GarageArea', 155],
                                 ['1stFlrSF', 148],
                                 ['LotFrontage', 140],
                                 ['BsmtFinSF1', 125],
                                 ['YearBuilt', 120],
                                 ['BsmtUnfSF', 120],
                                 ['YearRemodAdd', 100]],
                                columns=['feature', 'score']).sort_values('score')

        fig = px.bar(features, x="score", y="feature", orientation='h')
        fig.update_traces(marker_color='#048b9a', width=0.3)
        st.plotly_chart(fig, use_container_width=True)

    if choice == "Distributions":
        space(1)
        st.markdown('<body class="p">Distribuons les Distributions !</body>', unsafe_allow_html=True)

        space(1)
        st.markdown("En statistique, *la distribution statistique* ou *distribution des fréquences*, est un tableau "
                    "qui associe des classes de valeurs obtenues lors d'une expérience à leurs fréquences d'apparition."
                    " Ce tableau de valeurs est modélisé en théorie des probabilités par une loi de probabilité")

        space(2)
        st.title('Pourquoi parler de distribution ?')

        norm_df = norm.rvs(loc=data['SalePrice'].mean(), scale=data['SalePrice'].std(), size=20000)
        john_df = johnsonsu.rvs(a=-1.566133015864967, b=1.489964992053405,
                                loc=93994.94600320084, scale=55321.64688360436, size=10000)
        fig_norm = ff.create_distplot([data['SalePrice'], norm_df],
                                      ['SalePrice', 'Distribution Normale'],
                                      show_hist=False)
        fig_john = ff.create_distplot([data['SalePrice'], john_df],
                                      ['SalePrice', 'Distribution Johnsonsu'],
                                      show_hist=False)

        for fig in [fig_norm, fig_john]:
            fig.update_layout(font_family='IBM Plex Sans',
                              yaxis=dict(visible=False),
                              uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10),
                              legend=dict(x=1, y=1.02,
                                          orientation="h",
                                          yanchor="bottom",
                                          xanchor="right",
                                          bgcolor='rgba(0,0,0,0)',
                                          font=dict(size=12)))

        cols = st.columns(2)
        with cols[0]:
            st.subheader('Une distribution Normale')
            st.plotly_chart(fig_norm, use_container_width=True)
        with cols[1]:
            st.subheader('Une distribution Johnsonsu')
            st.plotly_chart(fig_john, use_container_width=True)

        space(2)
        st.title('Continuons sur les données... continues')

        space(1)
        cols = st.columns(2)
        with cols[0]:
            st.subheader('Surface du terrain')
            st.plotly_chart(fig_lotarea, use_container_width=True)
        with cols[1]:
            st.subheader("Largeur de l'accès à la route")
            st.plotly_chart(fig_lotfront, use_container_width=True)
        space(1)
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Surface au Rez-de-Chaussée")
            st.plotly_chart(fig_grliv, use_container_width=True)
        with cols[1]:
            st.subheader('Surface du 1er étage')
            st.plotly_chart(fig_florsf, use_container_width=True)
        space(1)
        # cols = st.columns(3)
        # with cols[0]:
        # st.subheader('Surface de la cave')
        # st.plotly_chart(fig_totbsmt, use_container_width=True)
        # with cols[1]:
        #   st.subheader('Surface de la cave habitable')
        #  st.plotly_chart(fig_bsmtfin, use_container_width=True)
        # with cols[2]:
        #   st.subheader('Surface de la cave non habitable')
        #  st.plotly_chart(fig_bsmtnofin, use_container_width=True)

        space(2)
        st.title('Des données pas si discrètes')
        temp = df.copy()
        for col in ['OverallQual', 'ExterQual', 'KitchenQual', 'GarageCars',
                    'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']:
            temp[col] = temp[col].astype(str)

        space(1)
        cols = st.columns(2)
        with cols[0]:
            st.subheader('Qualité Générale')
            fig = discret_layout(px.histogram(temp, x="OverallQual",
                                              marginal="box",
                                              category_orders=dict(OverallQual=df["OverallQual"].sort_values())))
            st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            st.subheader('Nombre de pièces')
            fig = discret_layout(px.histogram(temp, x="TotRmsAbvGrd",
                                              marginal="box",
                                              category_orders=dict(TotRmsAbvGrd=df["TotRmsAbvGrd"].sort_values())))
            fig.update_layout(font_family='IBM Plex Sans', uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10))
            st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.subheader('Qualité de la Cuisine')
            fig = discret_layout(px.histogram(temp, x="KitchenQual",
                                              marginal="box",
                                              category_orders=dict(KitchenQual=temp["KitchenQual"].sort_values())))
            fig.update_layout(font_family='IBM Plex Sans', uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10))
            st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            st.subheader("Qualité de l'Extérieur")
            fig = discret_layout(px.histogram(temp, x="ExterQual",
                                              marginal="box",
                                              category_orders=dict(ExterQual=temp["ExterQual"].sort_values())))
            fig.update_layout(font_family='IBM Plex Sans', uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10))
            st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        temp_bis = df[['FullBath', 'GarageCars']]
        temp_bis['FullBath'] = temp_bis['FullBath'].astype(str)
        temp_bis['GarageCars'] = temp_bis['GarageCars'].astype(str)

        with cols[0]:
            st.subheader('Nombre de Salles de Bain')
            fig = discret_layout(px.histogram(temp_bis, x="FullBath",
                                              marginal="box",
                                              category_orders=dict(FullBath=['0', '1', '2', '3'])))
            fig.update_layout(font_family='IBM Plex Sans', uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10))
            st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            st.subheader('Nombre de parkings')
            fig = discret_layout(px.histogram(temp_bis, x="GarageCars",
                                              marginal="box",
                                              category_orders=dict(GarageCars=['0', '1', '2', '3', '4'])))
            fig.update_layout(font_family='IBM Plex Sans', uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10))
            st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.subheader('Année de Construction')
            fig = discret_layout(px.histogram(temp, x="YearBuilt",
                                              marginal="box",
                                              category_orders=dict(YearBuilt=df["YearBuilt"].sort_values())))
            fig.update_layout(font_family='IBM Plex Sans', uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10))
            st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            st.subheader('Année de Rénovation')
            fig = discret_layout(px.histogram(temp, x="YearRemodAdd",
                                              marginal="box",
                                              category_orders=dict(YearRemodAdd=df["YearRemodAdd"].sort_values())))
            st.plotly_chart(fig, use_container_width=True)

    if choice == "Machine Learning":
        space(1)
        st.markdown('<body class="p">To Be or Not to Be Parametric...</body>', unsafe_allow_html=True)

        space(1)
        st.markdown("""
                    Les données sont passées au travers d'un outil de machine learning pour prédire les catégories 
                    de prix. Nous avons scindé les prix en tranches de *50 000€* et avons entrainé nos modèles dessus.
                    """)

        space(1)
        st.header("Deux familles de modèles")
        space(1)
        cols = st.columns(2)
        with cols[0]:
            st.subheader('Modèle Paramétrique :')
            st.markdown("""
                C'est un modèle d'apprentissage qui résume les données à un **ensemble de paramètres de taille fixe** 
                (indépendant du nombre d'exemples d'apprentissage). 

                Quelle que soit la quantité de données que vous  soumettez à un modèle paramétrique, 
                il ne changera pas d'avis sur le nombre de paramètres dont il a besoin.

                *- Artificial Intelligence: A Modern Approach, page 737*
                        """)
        with cols[1]:
            st.subheader('Modèle Non Paramétrique :')
            st.markdown("""
                        Les modèles non-paramétriques **ne font pas d'hypothèses fortes sur la forme de la donnée**. 
                        Ne faisant pas d'hypothèse sur la donnée, le modèle va cherchent à s'adapter au mieux aux 
                        données d'apprentissage en construisant une cartographie par itération,
                        tout en conservant une capacité de généralisation aux données non vues. 

                        En tant que telles, elles sont capables d'ajuster un grand nombre de formes de données.

                        *— Artificial Intelligence: A Modern Approach, page 757*           
                        """)

        space(2)
        st.header('Quel Modèle de Machine Learning ? Telle est la question')

        model_Param = pd.DataFrame([['MultinomialNB', 0.273972602739726],
                                    ['Perceptron', 0.3972602739726027],
                                    ['LogisticRegression', 0.5041095890410959],
                                    ['LinearDiscriminantAnalysis', 0.5534246575342465],
                                    ['GaussianNB', 0.5753424657534246]],
                                   columns=['model', 'score'])

        model_noParam = pd.DataFrame([['KNeighbors', 0.5150684931506849],
                                      ['DecisionTree', 0.5972602739726027],
                                      ['RandomForest', 0.6438356164383562],
                                      ['GradientBoosting', 0.6575342465753424],
                                      ['SVR', 0.7556555106068537]],
                                     columns=['model', 'score'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=model_Param.index,
                                 y=model_Param.score,
                                 text=model_Param.model,
                                 mode='lines+markers+text',
                                 name='Parametrique'))
        fig.add_trace(go.Scatter(x=model_noParam.index,
                                 y=model_noParam.score,
                                 text=model_noParam.model,
                                 mode='lines+markers+text',
                                 name='Non Parametrique'))
        fig.update_traces(textposition='top center')
        fig.update_layout(font_family='IBM Plex Sans',
                          xaxis=dict(visible=False),
                          uniformtext_minsize=10, uniformtext_mode='hide',
                          margin=dict(l=10, r=10, b=10, t=10),
                          legend=dict(x=1, y=1.02,
                                      orientation="h",
                                      yanchor="bottom",
                                      xanchor="right",
                                      bgcolor='rgba(0,0,0,0)',
                                      font=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True)

    if choice == "Clustering":
        space(1)
        st.markdown('<body class="p">Groupez les tous !</body>', unsafe_allow_html=True)
        space(2)
        st.write("Utilisation d'un modèle de clustering Birch :")
        space(1)
        data = load_df(
            "https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/train_clusters.csv")
        X = data.drop(columns=['ExterQual', 'KitchenQual'])

        BP = Birch(threshold=0.0001)

        img = "https://github.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/blob/main/Elbow%20Birch.png?raw=true"
        st.image(img, width=600)
        space(2)
        # Quick examination of elbow method to find numbers of clusters to make.
        # print('Elbow Method to determine the number of clusters to be formed:')
        # Elbow_M = KElbowVisualizer(BP, k=10)
        # Elbow_M = Elbow_M.fit(X)
        # fig = Elbow_M
        # fig = Elbow_M.show()
        # st.write(fig)

        # define the model
        model = Birch(threshold=0.01, n_clusters=5)
        # fit the model
        model.fit(X)
        # assign a cluster to each example
        yhat = model.predict(X)

        data['clusters'] = model.labels_

        fig = px.box(data, x="clusters", y="SalePrice", color="clusters")
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                           'paper_bgcolor': 'rgba(0,0,0,0)', })
        fig.update_layout(title='<b>Clusters Boxplot</b>',
                          title_x=0.5, title_font_family="Verdana", showlegend=False)
        # fig.update_yaxes(range=[0, 400000])

        st.plotly_chart(fig, use_container_width=True)

        clusters = data.groupby(by=['clusters']).count()['SalePrice']
        clusters = clusters.reset_index()
        clusters = clusters.rename(columns={'clusters': 'Clusters', 'SalePrice': 'Number of Houses'})

        fig = go.Figure(data=[go.Bar(
            x=clusters['Clusters'], y=clusters['Number of Houses'],
            text=clusters['Number of Houses'],
            textposition='auto',
        )])
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                           'paper_bgcolor': 'rgba(0,0,0,0)', })
        fig.update_layout(title='<b>Clusters size</b>',
                          title_x=0.5, title_font_family="Verdana", showlegend=False)
        fig.update_yaxes(title="Number of Houses")
        fig.update_xaxes(title="Clusters")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(data, x="GrLivArea", y="SalePrice", color="clusters")
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                           'paper_bgcolor': 'rgba(0,0,0,0)', })
        fig.update_layout(title='<b>Clusters</b>',
                          title_x=0.5, title_font_family="Verdana", showlegend=False)
        fig.update_yaxes(range=[0, 700000])
        fig.update_xaxes(range=[0, 4000])
        fig.update(layout_coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

if week == 'Semaine 3':
    choice = st.sidebar.radio("Semaine 3", ("Previously",
                                            "Data Regularization",
                                            "Data Scaling",
                                            "Appplication sur les Régressions",
                                            "Application sur les Classifications",
                                            "Conclusion"))

    if choice == 'Previously':
        space(1)
        st.markdown('''<body class="p">Previoulsy on the AMES'PROJECT !</body>''', unsafe_allow_html=True)

        space(2)
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Semaine 1 : ")

            st.markdown(
                '''
                - Exploration du dataset des maisons vendues à AIMES,
                - Recherche des corrélations entre les *features* présentes, 
                - Nettoyage des données,
                - Premier test de ML de prédiction du prix. 
                '''
            )

        with cols[1]:
            st.subheader("Semaine 2 : ")

            st.markdown(
                '''
                - Analyse de la distribution des datas,
                - Comparaison entre les outils de ML paramétriques et non paramétriques,
                - Entrainement d'un premier outil de clusturing.
                '''
            )

        space(1)
        st.title('What Next ?...')
        space(1)

        cols = st.columns(2)
        with cols[0]:
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3_6yJUAFXN1iH17Ey0MJcKAqNGqpks3HOn_1erR__8slAfWC33rDiziwjsSM7Pz7H4XY&usqp=CAU',
                     width=230)
            st.subheader('Amélioration des datas')
            st.markdown(
                '''
                1. Test de Régularisation
                2. Test de Scaling
                '''
            )
        with cols[1]:
            st.image('https://i.dlpng.com/static/png/5947133-icon-artificial-intelligence-201177-free-icons-library-artificial-intelligence-icon-1600_1600_preview.png',
                     width=215)
            st.write(' ')
            st.subheader('Améliorations des modèles de ML')
            st.markdown(
                '''
                1. Régression sur le Prix
                2. Classification du Bulding Type
                '''
            )

    if choice == 'Data Regularization':
        space(1)
        st.markdown('<body class="p">La Régularisation des données</body>', unsafe_allow_html=True)

        score_regul = {
            'Linear Regression': [80.50, 34099, 20587],
            'LassoCV': [80.44, 34143, 20609],
            'RidgeCV': [80.44, 34142, 20561],
            'ElasticNetCV': [79.80, 34686, 20683]
        }

        score_regul_df = pd.DataFrame.from_dict(
            score_regul, orient='index',
            columns=['R2', 'RMSE', 'MAE']
        )

        space(2)
        cols = st.columns(2)
        with cols[0]:
            st.subheader('Score R2')
            fig = go.Figure()
            fig.add_bar(x=score_regul_df.index, y=score_regul_df['R2'],
                        text=score_regul_df['R2'],
                        textposition='auto'
                        )
            fig.update_layout(yaxis=dict(range=[0, 100]),
                              margin=dict(l=10, r=10, b=2, t=47),
                              )
            fig.update_xaxes(title_text='(Higher is better)')
            st.plotly_chart(fig, use_container_width=True)

        with cols[1]:
            st.subheader('Score RMSE et MAE')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=score_regul_df.index,
                                     y=score_regul_df.RMSE,
                                     text=score_regul_df.index,
                                     mode='lines+markers+text',
                                     name='RMSE'))
            fig.add_trace(go.Scatter(x=score_regul_df.index,
                                     y=score_regul_df.MAE,
                                     text=score_regul_df.index,
                                     mode='lines+markers+text',
                                     name='MAE'))
            fig.update_traces(textposition='top center')
            fig.update_layout(font_family='IBM Plex Sans',
                              yaxis=dict(range=[15000, 40000]),
                              uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10),
                              legend=dict(x=1, y=1.02,
                                          orientation="h",
                                          yanchor="bottom",
                                          xanchor="right",
                                          bgcolor='rgba(0,0,0,0)',
                                          font=dict(size=12)))
            fig.update_xaxes(title_text='(Lower is better)')
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Détail des statistiques", expanded=False):
            st.dataframe(score_regul_df)

        space(3)
        st.subheader("L'impact de la Régularisation sur les coéfficients des features")
        space(2)

        coef = pd.DataFrame([[-1.57718011e+04, -2.10435130e+04, -2.03652910e+04, -4.41646475e+03,
                               9.81179490e+03, -1.63959543e+03, 1.75892799e+04, -1.74179672e+04,
                               -4.76797889e+03, -1.71500061e+04, -1.93095272e+04, -7.28056503e+03,
                               -6.73851754e+03, -1.57770778e+04, -6.96620276e+03, 5.99830083e+04,
                               3.94455790e+04, -1.98583176e+04, -1.57810013e+04, -4.76658600e+03,
                               -8.15131628e+03, 8.04924668e+03, 4.77423124e+04, 4.55467127e+03,
                               1.86528582e+04, -5.74323165e+04, -1.73879107e+04, -1.69562767e+04,
                               -9.98704406e+03, -4.04544540e+04, -2.10901544e+04, 5.80492412e-01,
                               -6.35719889e+01, 5.81487143e+00, -7.25756856e+00, 1.31239040e+04,
                               3.56064776e+01, 6.58398144e+03, 1.21886530e+04, 1.22587877e+04,
                               1.93438658e+01, 1.18976632e+01, 6.22625576e+00, 5.61740879e+02,
                               1.25875373e+03, 2.13837902e+01, 1.05628292e+02],
                                [-1.58679113e+04, -1.66210789e+04, -1.94350642e+04, -3.86581331e+03,
                                9.28602753e+03, -2.14921088e+03,  1.71219092e+04, -1.69771131e+04,
                               -5.32513678e+03, -1.60804205e+04, -1.82079381e+04, -7.32519653e+03,
                               -6.70781740e+03, -1.46158166e+04, -7.08069778e+03,  5.84533288e+04,
                                3.84184594e+04, -1.92478425e+04, -1.48313835e+04, -4.72372768e+03,
                               -8.46782464e+03,  7.42664726e+03,  4.57553379e+04,  3.79511029e+03,
                                1.72731736e+04, -4.87495492e+04, -1.51348750e+04, -1.44294216e+04,
                               -7.64085867e+03, -3.48886840e+04, -1.87958365e+04,  5.82771747e-01,
                               -5.95106152e+01,  6.03363329e+00, -7.10941040e+00,  1.32775921e+04,
                                3.55196800e+01,  6.82909507e+03,  1.21623537e+04,  1.19089814e+04,
                                1.79858257e+01,  1.18509541e+01,  6.34543180e+00,  6.04645820e+02,
                                1.31311835e+03,  3.24319439e+01,  1.09916992e+02],
                             ['Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale',
                                'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr',
                                'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert',
                                'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel',
                                'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes',
                                'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown',
                                'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW',
                                'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber',
                                'Neighborhood_Veenker', 'GarageType_2Types', 'GarageType_Attchd',
                                'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort',
                                'GarageType_Detchd', 'LotArea', 'LotFrontage', 'BsmtFinSF1',
                                'BsmtUnfSF', 'OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual',
                                'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
                                'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
                             ]).T
        coef.rename(columns={0:'Lasso', 1:'Ridge', 2:'Feature'}, inplace=True)
        coef.set_index('Feature', inplace=True)
        coef['Différence'] = ((coef['Ridge']-coef['Lasso']) / (coef['Lasso']))*100

        cols = st.columns(2)
        with cols[0]:
            st.markdown(f'''<p class="value">{len(coef[coef['Lasso']==0])}</p>
                            <p class="titre">Nombre de Coefficients éliminés par Lasso</p>
                         ''', unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f'''<p class="value">{round(coef['Différence'].median(), 2)}%</p>
                            <p class="titre">Médiane de la différence des coefficients entre Lasso et Ridge</p>
                         ''', unsafe_allow_html=True)

        space(1)
        with st.expander("Détail des statistiques", expanded=False):
            st.dataframe(coef)

    if choice == 'Data Scaling':
        space(1)
        st.markdown('<body class="p">Le Scaling des données</body>', unsafe_allow_html=True)

        space(2)
        st.title("Le Scaling en théorie")
        cols = st.columns(2)
        with cols[0]:
            st.subheader('La Distribution des datas')
            norm_df = norm.rvs(loc=data['1stFlrSF'].mean(), scale=data['1stFlrSF'].std(), size=20000)
            fig_norm = ff.create_distplot([data['1stFlrSF'], norm_df],
                                          ['Surface', 'Distribution Normale'],
                                          show_hist=False)
            fig_norm.update_layout(font_family='IBM Plex Sans',
                              yaxis=dict(visible=False),
                              uniformtext_minsize=10, uniformtext_mode='hide',
                              margin=dict(l=10, r=10, b=10, t=10),
                              legend=dict(x=1, y=1.02,
                                          orientation="h",
                                          yanchor="bottom",
                                          xanchor="right",
                                          bgcolor='rgba(0,0,0,0)',
                                          font=dict(size=12)))
            st.plotly_chart(fig_norm, use_container_width=True)

        with cols[1]:
            st.subheader("La présence d'outliers")
            fig = px.box(data, y=["1stFlrSF"])
            fig.update_layout(font_family='IBM Plex Sans',
                              margin=dict(l=10, r=10, b=10, t=50),
                              xaxis=dict(title='Surface'),
                              yaxis=dict(title=''))
            fig.update_yaxes(title_text='Foot Square')
            st.plotly_chart(fig, use_container_width=True)

        space(2)
        st.title("Le Scaling en pratique")

        option = st.multiselect('Action de Scaling : ',
                       ['Sans Scaling',
                        'Scaling des Features Qualitatives',
                        'Retrait des Outliers',
                        "Application d'un StandardScaler"],
                       ['Sans Scaling']
                       )
        scaling_score = pd.DataFrame([[34142, 33310, 23370, 23370],
                                      [20609, 19227, 16132, 16113],
                                      ['Sans Scaling',
                                       'Scaling des Features Qualitatives',
                                       'Retrait des Outliers',
                                       "Application d'un StandardScaler"]
                                      ]).T
        scaling_score.rename(columns={0:'RMSE', 1:'MAE', 2:'Scaler'}, inplace=True)
        scaling_score.set_index('Scaler', inplace=True)
        data = scaling_score.loc[option]

        space(1)
        st.subheader('Scores du Ridge')
        fig = go.Figure(data=[
            go.Bar(name='RMSE',
                   x=data.index,
                   y=data['RMSE'],
                   text=data['RMSE'],
                   textfont=dict(
                       size=18),
                   ),
            go.Bar(name='MAE',
                   x=data.index,
                   y=data['MAE'],
                   text=data['MAE'],
                   textfont=dict(
                       size=18),
                   )
        ])
        fig.update_layout(margin=dict(l=10, r=10, b=10, t=10),
                          legend=dict(x=1, y=1.02,
                                      orientation="h",
                                      yanchor="bottom",
                                      xanchor="right",
                                      bgcolor='rgba(0,0,0,0)',
                                      font=dict(size=12)))
        fig.update_layout(barmode='group')
        fig.update_xaxes(title_text='(Lower is better)')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Détail des scores", expanded=False):
            data['Evolution'] = data['RMSE']-data['MAE']
            st.dataframe(data)

    if choice == 'Appplication sur les Régressions':
        space(1)
        st.markdown("<body class='p'>Généralisation des Optimisations</body>", unsafe_allow_html=True)

        space(2)
        st.subheader("Un Scaling non généralisable")

        models_scoring = pd.DataFrame([
            ['Ridge', 23370, 23370],
            ['SVM', 32760, 66145],
            ['Neural Net', 46363, 89300],
            ['Decision Tree', 39448, 40397],
            ['Random Forest', 25582, 25398],
            ['Nearest Neighbors', 41279, 64267],
            ['Gradient Boosting', 24473, 24439],],
            columns=['Model', 'Sans StandardScaler', 'Avec StandardScaler']
        ).set_index('Model')

        fig = go.Figure(data=[
            go.Bar(name='Sans StandardScaler',
                   x=models_scoring.index,
                   y=models_scoring['Sans StandardScaler'],
                   text=models_scoring['Sans StandardScaler'],
                   textfont=dict(
                       size=18),
                   ),
            go.Bar(name='Avec StandardScaler',
                   x=models_scoring.index,
                   y=models_scoring['Avec StandardScaler'],
                   text=models_scoring['Avec StandardScaler'],
                   textfont=dict(
                       size=18),
                   )
        ])
        fig.update_layout(margin=dict(l=10, r=10, b=10, t=10),
                          legend=dict(x=1, y=1.02,
                                      orientation="h",
                                      yanchor="bottom",
                                      xanchor="right",
                                      bgcolor='rgba(0,0,0,0)',
                                      font=dict(size=12)))
        fig.update_layout(barmode='group')
        fig.update_yaxes(title_text='RMSE')
        fig.update_xaxes(title_text='(Lower is better)')
        st.plotly_chart(fig, use_container_width=True)

        space(2)
        st.title('Les finalistes')
        space(2)

        st.subheader("Gradient Boosting")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f'''<p class="sub_value">88,31%</p>
                            <p class="text">R2</p>
                         ''', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'''<p class="sub_value">23 726 $</p>
                            <p class="text">RMSE</p>
                         ''', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'''<p class="sub_value">16 183 $</p>
                            <p class="text">MAE</p>
                         ''', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'''<p class="sub_value">10,04%</p>
                            <p class="text">MAPE</p>
                         ''', unsafe_allow_html=True)
        space(1)
        st.subheader("Ridge")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f'''<p class="sub_value">88,65%</p>
                            <p class="text">R2</p>
                         ''', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'''<p class="sub_value">23 366 $</p>
                            <p class="text">RMSE</p>
                         ''', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'''<p class="sub_value">16 098 $</p>
                            <p class="text">MAE</p>
                         ''', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'''<p class="sub_value">9,98%</p>
                            <p class="text">MAPE</p>
                         ''', unsafe_allow_html=True)

        space(3)
        st.title('Le plus efficient')
        space(1)

        cols = st.columns([2, 3])
        with cols[0]:
            score_type = st.radio('Selection du type de score :', ['Score Brut', 'Efficience'])

            if score_type == 'Efficience':
                space(2)
                st.subheader("Temps d'execution/boucle :")
                st.markdown('''<p class="text">Ridge : 2,5 secondes</p>
                             ''', unsafe_allow_html=True)
                st.markdown(f'''<p class="text">Gradien Boosting : 32 secondes</p>
                                 ''', unsafe_allow_html=True)


        with cols[1]:
            if score_type == 'Score Brut':
                score_RG = pd.DataFrame(
                    [
                        ['Ridge', 23366, 16098],
                        ['Gradian Boosting', 23726, 16183],
                    ],
                    columns=['Model', 'RMSE', 'MAE']
                ).set_index('Model')
                title='(The lower is better)'

            if score_type == 'Efficience':
                score_RG = pd.DataFrame(
                    [
                        ['Ridge', 9163, 6309],
                        ['Gradian Boosting', 741, 506],
                    ],
                    columns=['Model', 'RMSE', 'MAE']
                ).set_index('Model')
                title = '(The higher is better)'

            fig = go.Figure(data=[
                go.Bar(name='RMSE',
                       x=score_RG.index,
                       y=score_RG['RMSE'],
                       text=score_RG['RMSE'],
                       textfont=dict(
                           size=18),
                       ),
                go.Bar(name='MAE',
                       x=score_RG.index,
                       y=score_RG['MAE'],
                       text=score_RG['MAE'],
                       textfont=dict(
                           size=18),
                       )
            ])
            fig.update_layout(margin=dict(l=10, r=10, b=10, t=10),
                              legend=dict(x=1, y=1.02,
                                          orientation="h",
                                          yanchor="bottom",
                                          xanchor="right",
                                          bgcolor='rgba(0,0,0,0)',
                                          font=dict(size=12)))
            fig.update_layout(barmode='group')
            fig.update_xaxes(title_text=title)
            st.plotly_chart(fig, use_container_width=True)

    if choice == 'Application sur les Classifications':
        space(1)
        st.markdown("<body class='p'>Premiers essais de Classification</body>", unsafe_allow_html=True)

        space(2)
        st.write(
            "Pour aller plus loin dans la démarche, nous avons voulu entrainer un modèle de machine learning pour permettre une classification des maisons en fonction du type de résidence.")
        df = transfo(
            pd.read_csv('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Projet-1-DataScience/main/train.csv'))

        X = df.drop(columns=['BldgType'])
        y = df['BldgType']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = [RidgeClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(),
                  LogisticRegression(C=0.001, penalty='l2')]
        models_name = ['Ridge', 'Decision Tree', 'KNeighbors', 'LogisticRegression']

        best_model = {}
        for model, name in zip(models, models_name):
            score = cross_val_score(model, X, y, cv=5).mean()
            best_model[name] = score

        model_classique = pd.DataFrame.from_dict(best_model,
                                                 orient='index',
                                                 columns=['score']).sort_values(by='score',
                                                                                ascending=False)
        model_classique.reset_index(inplace=True)

        st.subheader("Comparatif des scores d'Accuracy")
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                value(round(model_classique.iloc[i,1]*100, 2))
                st.markdown(f'''<p class="text">{model_classique.iloc[i,0]}</p>
                             ''', unsafe_allow_html=True)

        space(1)
        st.write("Suite au calcul des scores nous avons décidé d'appliquer un modèle de régression logistique")
        space(3)

        st.subheader("Les scores ROC/AUC par classe")
        # Fit the model
        model = LogisticRegression()
        model.fit(X, y)
        y_scores = model.predict_proba(X)

        # One hot encode the labels in order to plot them
        y_onehot = pd.get_dummies(y, columns=model.classes_)

        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()

        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            # yaxis=dict(scaleanchor="x", scaleratio=1),
            # xaxis=dict(constrain='domain'),
            height=700

        )
        fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',
                          margin=dict(l=70, r=70, b=70, t=10),
                          legend=dict(x=1, y=1.02,
                                      orientation="h",
                                      yanchor="bottom",
                                      xanchor="right",
                                      bgcolor='rgba(0,0,0,0)',
                                      font=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True)

    if choice == 'Conclusion':
        space(2)
        cols = st.columns([1, 4, 1])
        with cols[1]:
            st.image("http://bribriange49.b.r.pic.centerblog.net/f7f71790.gif", width=600)

        space(3)
        cols = st.columns([3, 1, 3])
        with cols[1]:
            st.subheader('La Team')
        cols = st.columns([1, 2, 2, 2, 2])
        names = ['Matthieu', 'Mehdi', 'Michael', 'Sebastien']
        for i, name in zip(range(1, 5), names):
            with cols[i]:
                st.markdown(f'''<p class="text">{name}</p>
                             ''', unsafe_allow_html=True)





