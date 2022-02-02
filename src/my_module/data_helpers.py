from functools import partial
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import random
from sklearn import decomposition, preprocessing, model_selection, linear_model, neighbors, metrics, datasets, impute
from sklearn.base import ClassifierMixin
from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, accuracy_score, mean_squared_error, r2_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from hyperopt import fmin, tpe, hp, anneal, Trials, space_eval
from imblearn.over_sampling import SMOTE
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer

# Affiche la taille du jeu de données
def displayDataShape(message, data):
    shape = data.shape
    print(f"{message} : {shape[0]} lignes et {shape[1]} colonnes\n")

# Affichage du % de valeurs manquantes par colonne
def displayNanPercent(data):
    # Total missing values
    mis_val = data.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * data.isnull().sum() / len(data)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Valeurs manquantes', 1 : '% de valeurs manquantes'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% de valeurs manquantes', ascending=False).round(1)

    # Print some summary information
    print ("Le jeu de données contient " + str(data.shape[1]) + " colonnes.\n"
           + str(mis_val_table_ren_columns.shape[0]) +
           " colonnes ont des valeurs manquantes.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
# Affiche les diagrammes en camembert des variables qualitatives

def drawPieplot(data, columns, dims_fig):
    nbr_rows = int(len(columns)/2) + 1
    index = 1

    for column in columns:
        values_name = data[column].value_counts().index
        data_sum = data[column].value_counts()
        iteration = 0
        for value in values_name:
            if iteration >= 30:
                break
            elif data_sum[value]/data_sum.sum()*100 >= 1.5:
                iteration+= 1
            else:
                break

        data_bis = data[column].where(data[column].isna() | data[column].isin(data[column].value_counts().index[:iteration]), other='other')

        if 'other' not in data_bis.value_counts().index:
            plt.subplot(nbr_rows, 2, index)
            data_bis.value_counts().plot.pie(figsize=dims_fig,
                                             autopct='%1.1f%%',
                                             startangle = 60,
                                             rotatelabels = True,
                                             pctdistance = 0.85)
            plt.ylabel("Nombre de prêts")
            plt.title(f"Répartition des prêts par {column}", pad=50)
            index += 1
        elif data_bis.value_counts()['other']/data_bis.value_counts().sum()*100 < 60:
            plt.subplot(nbr_rows, 2, index)
            data_bis.value_counts().plot.pie(figsize=dims_fig,
                                             autopct='%1.1f%%',
                                             startangle = 60,
                                             rotatelabels = True,
                                             pctdistance = 0.85)
            plt.ylabel("Nombre de prêts")
            plt.title(f"Répartition des prêts par {column}", pad=50)
            index += 1
    plt.show

# Affiche les histogrammes et les boîtes à moustaches de chaque variable quantitative
def drawHistAndBoxPlot(data, columns, dims_fig):
    nbr_rows = int(len(columns))
    index = 1
    plt.figure(figsize=dims_fig)
    for column in columns:

        plt.subplot(nbr_rows, 2, index)
        plt.hist(data[column])
        plt.xlabel(f"{column}")
        plt.ylabel("Nombre de prêts")
        plt.title(f"Histogramme - {column}")
        index += 1

        plt.subplot(nbr_rows, 2, index)
        sns.boxplot(x=data[column])
        plt.xlabel(column)
        plt.title(f"Boite à moustaches pour {column}")
        index += 1
    plt.show()

def analyse_num_with_target_bin(positive_df,negative_df,target_var,columns,remove=["Col_ID"]):
    num_cols = [col for col in positive_df.columns if (not col in positive_df.select_dtypes(["object","category"]).columns.to_list() and col in columns)]
    if len(remove) > 0:
        num_cols = [col for col in num_cols if (col not in remove)]

    for col in num_cols:
        fig,axes = plt.subplots(1,3,figsize=(25,6))
        sns.distplot(positive_df[col],label="positive",ax=axes[0])
        sns.distplot(negative_df[col],label="negative",ax=axes[0])

        pd.concat([positive_df,negative_df],axis=0).boxplot(str(col),by=target_var,ax=axes[1],vert=False)

        pd.concat([positive_df,negative_df],axis=0).boxplot(str(col),ax=axes[2],vert=False)

        axes[0].legend()
        # Taille max du nom de la colonne 50
        axes[0].set_title(col[:50] + " | Distplot")
        axes[1].set_title(col[:50] + " | Boxplot split")
        axes[2].set_title(col[:50] + " | Boxplot global")

        axes[0].set_xlabel("")
        axes[1].set_xlabel("")
        axes[2].set_xlabel("")

        axes[2].set_yticklabels([])
        axes[2].set_yticks([])

        plt.show()

# One-hot encoding pour les variables catégorielles
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(bureau, bb):

    bb, bb_cat = one_hot_encoder(bb)
    bureau, bureau_cat = one_hot_encoder(bureau)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(prev):
    prev, cat_cols = one_hot_encoder(prev)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(pos):
    pos, cat_cols = one_hot_encoder(pos)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(ins):
    ins, cat_cols = one_hot_encoder(ins)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(cc):
    cc, cat_cols = one_hot_encoder(cc)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# Détermine les composantes principales
# Trace les éboulis des valeurs propres
# Affiche la projection des individus sur les différents plans factoriels
# Affiche les cercles de corrélations
def acpAnalysis(data, target):
    n = data.shape[0]
    p = data.shape[1]

    # On instancie l'object ACP
    acp = decomposition.PCA(svd_solver='full')
    # On récupère les coordonnées factorielles Fik pour chaque individu (projection des individus sur les composantes principales)
    coord = acp.fit_transform(data)

    # Création d'un Datframe contenant les coordonnées factiorelles, le nom de chaque produit et le nom des colonnes correspondant à chaque composante principale
    projected_data = pd.DataFrame(
        data=coord,
        index=data.index,
        columns=[ f'F{i}' for i in range(1, p+1) ]
    )
    # On rajoute la colonne TARGET
    projected_data['TARGET'] = target

    # valeur de la variance corrigée
    eigval = (n-1)/n*acp.explained_variance_
    eigval_ratio = (n-1)/n*acp.explained_variance_ratio_

    # On affiche l'éboulis des valeurs propres
    plt.plot(np.arange(1,p+1),eigval,c="red",marker='o')
    plt.title("Eboulis des valeurs propres")
    plt.ylabel("Valeur propre")
    plt.xlabel("Rang de l'axe d'inertie")
    plt.show()

    plt.bar(np.arange(1, p+1), eigval_ratio*100)
    plt.plot(np.arange(1, p+1), eigval_ratio.cumsum()*100,c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

    # On détermine le nombre de composantes à analyser
    # On ne considère pas comme importants les axes dont l'inertie associée est inférieue à (100/p)%
    k = np.where(eigval_ratio < 1/p)[0][0]

    if (k % 2) != 0:
        k -= 1

    print (f"Le nombre de composantes à analyser est de {k}")

    # racine carrée des valeurs propres
    sqrt_eigval = np.sqrt(eigval)

    # corrélation des variables avec les axes
    corvar = np.zeros((p,p))
    for j in range(p):
        corvar[:,j] = acp.components_[j,:] * sqrt_eigval[j]

    for i in range(0, 2, 2):
        # --------------- projection des individus dans un plan factoriel ---------------
        fig, axes = plt.subplots(1, 2, figsize=(24,12))
        axes[0].set_xlim(projected_data[f'F{i+1}'].min(),projected_data[f'F{i+1}'].max())
        axes[0].set_ylim(projected_data[f'F{i+2}'].min(),projected_data[f'F{i+2}'].max())
        sns.scatterplot(
            ax=axes[0], x=f'F{i+1}',
            y=f'F{i+2}',
            data=projected_data,
            hue="TARGET")

        axes[0].set_xlabel(f'F{i+1}')
        axes[0].set_ylabel(f'F{i+2}')
        axes[0].set_title(f"Projection des individus sur 'F{i+1}' et 'F{i+2}'")

        # ajouter les axes
        axes[0].plot([projected_data[f'F{i+1}'].min(),projected_data[f'F{i+1}'].max()],[0,0],color='silver',linestyle='--',linewidth=3)
        axes[0].plot([0,0],[projected_data[f'F{i+2}'].min(),projected_data[f'F{i+2}'].max()],color='silver',linestyle='--',linewidth=3)

        # --------------- cercle des corrélations ---------------
        axes[1].set_xlim(-1,1)
        axes[1].set_ylim(-1,1)

        # affichage des étiquettes (noms des variables)
        for j in range(p):
            axes[1].annotate(data.columns[j],(corvar[j,i],corvar[j,i+1]))
            axes[1].arrow(0, 0, corvar[j,i], corvar[j,i+1], length_includes_head=True, head_width=0.04)

        # ajouter un cercle
        cercle = plt.Circle((0,0),1,color='blue',fill=False)
        axes[1].add_artist(cercle)

        # ajouter les axes
        axes[1].plot([-1,1],[0,0],color='silver',linestyle='--',linewidth=3)
        axes[1].plot([0,0],[-1,1],color='silver',linestyle='--',linewidth=3)

        # nom des axes, avec le pourcentage d'inertie expliqué
        axes[1].set_xlabel('F{} ({}%)'.format(i+1, round(100*acp.explained_variance_ratio_[i],1)))
        axes[1].set_ylabel('F{} ({}%)'.format(i+2, round(100*acp.explained_variance_ratio_[i+1],1)))
        axes[1].set_title(f"Cercle des corrélations (F{i+1} et F{i+2})")

        # affichage
        plt.show()


def hyperopt_function(space, classifier, X, y, cv, sm, smote):

    scores = []

    for train_idx, test_idx, in cv.split(X, y):
        X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
        X_val_cv, y_val_cv = X.iloc[test_idx], y.iloc[test_idx]

        if smote:
            X_train_cv, y_train_cv = sm.fit_resample(X_train_cv, y_train_cv)

        classifier.set_params(**space)
        classifier.fit(X_train_cv, y_train_cv)

        tn, fp, fn, tp = confusion_matrix(y_val_cv, classifier.predict(X_val_cv)).ravel()

        score = (0.9*fn + 0.1*fp)/y.size

        scores.append(score)

    return sum(scores)/len(scores)

def find_best_parameters(classifier: ClassifierMixin, space, X, y, X_test, y_test, smote=True):
    start_training_time = time()
    trials = Trials()

    cv = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    sm = SMOTE(random_state=42)

    fmin_function = partial(hyperopt_function, classifier=classifier, X=X, y=y, cv=cv, sm=sm, smote=smote)

    best = space_eval(space, fmin(fn=fmin_function, space=space, algo=tpe.suggest, trials=trials, max_evals=10))

    training_time = time() - start_training_time

    classifier.set_params(**best)

    if smote:
        X_smote, y_smote = sm.fit_resample(X, y)
        classifier.fit(X_smote, y_smote)
    else :
        classifier.fit(X, y)
        
    start_predict_time = time()
    y_pred = classifier.predict(X_test)
    predict_time = time() - start_predict_time

    if hasattr(classifier, "predict_proba"):
        y_pred_proba = classifier.predict_proba(X_test)[:,1]
    elif hasattr(classifier, "decision_function"):
        y_pred_proba = classifier.decision_function(X_test)
    else:
        y_pred_proba = y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    personnalised_metric = (0.9*fn + 0.1*fp)/y.size

    return {
        "Classifier": classifier,
        "Best params": best,
        "Predict time": predict_time,
        "Training time": training_time,
        "Personnalised metric": personnalised_metric,
        "Confusion matrix": confusion_matrix(y_test, y_pred),
        "f1 score": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Average precision": average_precision_score(y_test, y_pred_proba)
    }

def plot_classifier_results(classifier, X, y):

    fig, ax = plt.subplots(1,3,figsize=(24,8))

    plot_confusion_matrix(estimator=classifier, X=X, y_true=y, cmap='BuPu_r', ax=ax[0])
    plot_precision_recall_curve(estimator=classifier, X=X, y=y, ax=ax[1])
    plot_roc_curve(estimator=classifier, X=X, y=y, ax=ax[2])

    plt.show()

def lime(classifier, index, X_train, X_test, y_test):

    explainer = LimeTabularExplainer(
        X_train.to_numpy(),
        feature_names=X_train.columns
        )

    exp = explainer.explain_instance(
        X_test.iloc[index],
        classifier.predict_proba
    )

    print(f"Vraie valeur : {y_test.iloc[index]}")
    exp.show_in_notebook()