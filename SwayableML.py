import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#
# Get Survey Responses
import bson
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
#
df_new = pd.read_csv('df_complete.csv')
#
#
# Get Responses
bson_file_responses = open('responses.bson', 'rb')
bson_docs_responses = bson.decode_all(bson_file_responses.read())
df_responses = pd.DataFrame(bson_docs_responses)
#
# Get Content
bson_file_content = open('content2.bson', 'rb')
bson_docs_content = bson.decode_all(bson_file_content.read())
df_content = pd.DataFrame(bson_docs_content)
df_content.to_csv('df_content.csv')
#
# Drop unnamed column
df_new.drop(['Unnamed: 0'], axis=1, inplace=True)
#
# Remove Unrealistic Annual Incomes
df_new['AnnualIncome'] = df_new['AnnualIncome'].mask(df_new['AnnualIncome'] < 0)
df_new['AnnualIncome'] = df_new['AnnualIncome'].mask(df_new['AnnualIncome'] > 200000)
df_new['survey_id'] = df_responses['survey_id']
#
# Convert Education to string
df_new['Education'].astype(str)
#
# Calculate Ages and Remove Unrealistic Values
df_new['Age'] = 2018 - df_new['BirthYear']
df_new['Age'] = df_new['Age'].mask(df_new['Age'] > 90)
#
# Replace 'none' political beliefs to 5 (middle of range)
df_new['PoliticalBeliefs'].replace(to_replace='none', value=int(5), regex=True, inplace = True)
df_new['PoliticalBeliefs'].fillna(5, inplace = True)
df_new['PoliticalBeliefs'].astype(str).astype('int64')
#
# Replace 'none' political party to 5 (middle of range)
df_new['PoliticalParty'].replace(to_replace='none', value=int(5), regex=True, inplace = True)
df_new['PoliticalParty'].fillna(5, inplace = True)
df_new['PoliticalParty'].astype(str).astype(int)
#
# Replace 'none' ViewsOnRace to 5 (middle of range)
df_new['ViewsOnRace'].replace(to_replace='none', value=int(5), regex=True, inplace = True)
df_new['ViewsOnRace'].fillna(5, inplace = True)
df_new['ViewsOnRace'].astype(str).astype(int)
#
# Replace 'none' TrupApprove to 5 (middle of range)
df_new['TrumpApprove'].replace(to_replace='none', value=int(5), regex=True, inplace = True)
df_new['TrumpApprove'].fillna(5, inplace = True)
df_new['TrumpApprove'].astype(str).astype(float)
#
# Convert Zip Codes
df_new['ZipCodeNew'] = df_new['ZipCode'].where(df_new['ZipCode'].str.len() == 5, 
                                               df_new['ZipCode'].str[:5])
df_new['ZipCodeNew'].astype(str)
df_new['ZipCodeNew'].replace('no on', np.NaN)
pd.to_numeric(df_new['ZipCodeNew'], downcast = 'integer', errors = 'coerce')
#
# Calculate Viewer Agreement Scores for each Viewer
df_new['Score'] = df_new[['Persuasion', 'Agreement', 'BrandLift', 
      'All: DirectVideoReaction', 'OpinionLift', 'DirectVideoReaction', 
      'Direct Video Reaction', 'DirectReaction', 'Support', 'Brand Lift',
      '18-35 In District: DirectVideoReaction', '18-35: DirectVideoReaction',
      '18-35: VideoReaction', 'Likely Voters: VideoReaction', 'Activation',
      'CandidateAndPartySupport', 'DemocraticPartySupport', 'GlobalWarming',
      'TaxPlan', 'CandidateSupport', 'Brand Issue Narrative', 'Greens',
      'ImpeachTrump', 'IncreaseSupportForChoice', 'OpposeTrumpCourtNomination',
      'Support for Prop 10']].mean(axis = 1)
#
# Get rid of rows with no test results/scores
hasresults = df_responses['analysis']
hasresults.dropna(inplace=True)
#
# Create table of significant predictor values
df_test = df_new[['_id', 'survey_id', 'content_id', 'Sex', 'AnnualIncome', 'Age', 'Education', 'Ethnicity', 'PoliticalBeliefs', 
                  'PoliticalParty', 'ViewsOnRace', 'TrumpApprove', 'Score']] # dropped zipcodenew
#
# df_test2 used for individual prediction
#df_test2 = df_new[['_id', 'survey_id', 'content_id', 'Sex', 'AnnualIncome', 'Age', 'Education', 'Ethnicity', 'PoliticalBeliefs', 
#                  'PoliticalParty', 'ViewsOnRace', 'TrumpApprove', 'Score']]
#
# Fill in missing income values with median
income_median = df_test['AnnualIncome'].median()
def income_calc(cols):
    """ This function fills in missing incomes with the mean income"""
    income = cols[0]
    if pd.isnull(income):
        return income_median
    else:
        return income

df_test['AnnualIncome'] = df_test[['AnnualIncome']].apply(income_calc, axis=1) 

# Fill in missing age values with median
age_median = df_test['Age'].median()
def age_calc(cols):
    """ This function fills in missing ages with the median age """
    age = cols[0]
    if pd.isnull(age):
        return age_median
    else:
        return age
#
df_test['Age'] = df_test[['Age']].apply(age_calc, axis=1)   
#
df_test.dropna(subset = ['Score', 'Sex'], inplace = True)
#
# Import data test categorical columns - sex
sex_id = pd.get_dummies(df_test['Sex'], drop_first = True)
df_test.drop(['Sex'], axis=1, inplace=True)

## Import data test categorical columns - survey_id
serv_id = pd.get_dummies(df_test['survey_id'], drop_first = True)
df_test.drop(['survey_id'], axis=1, inplace=True)

# Import data test categorical columns - education
edu_id = pd.get_dummies(df_test['Education'], drop_first = True)
df_test.drop(['Education'], axis=1, inplace=True)

# Import data test categorical columns - ethnicity
eth_id = pd.get_dummies(df_test['Ethnicity'], drop_first = True)
df_test.drop(['Ethnicity'], axis=1, inplace=True)
#
df_all = pd.concat([df_test, sex_id, serv_id, eth_id, edu_id], axis=1)
#
#
###############################################################################
#                         ANALYSIS USING TEST VIDEOS
###############################################################################
from sklearn.feature_extraction.text import CountVectorizer
df_vid = []
df_1 = []
#
# Read in extracted video features
df_vid = pd.read_csv('Video_features_updated.csv')
df_vid.drop(df_vid.columns[0], axis=1, inplace = True)
df_vid.drop(['title'], axis=1, inplace = True)
df_vid.drop_duplicates(inplace = True)
df_vid = df_vid.reset_index(drop=True)
#
#
df_full = pd.merge(df_all, df_vid, how='inner', left_on='content_id', right_on = '_id')
#df_full.dropna(inplace = True)
#
# Extract Word Data (spoken)
word_dict = {'trump': 0, 'abortion': 1, 'climate': 2, 'working': 3, 'vote': 4, 
             'president': 5, 'parenthood': 6, 'health': 7, 'school': 8, 
             'states': 9, 'student': 10, 'world': 11, 'american': 12,
             'change': 13, 'country': 14, 'good': 15, 'government': 16, 
             'great': 17, 'love': 18, 'money': 19, 'new': 20, 'planned': 21,
             'teaching': 22, 'think': 23, 'time': 24, 'today': 25, 'years': 26,
             'children': 27, 'care': 28, 'important': 29, 'need': 30, 
             'police': 31, 'tax': 32, 'share': 33, 'veteran': 34, 'voted': 35, 
             'want': 36, 'jobs': 37, 'coal': 38, 'politician': 39, 
             'politicians': 40, 'corporations': 41, 'corporation': 42, 
             'diverse': 43, 'diversity': 44, 'black': 45, 'court': 46, 
             'federal': 47, 'community': 48, 'communities': 49, 'I': 50, 
             'you': 51, 'us': 52, 'our': 53, 'we': 54, 'me': 55, 'my': 56, 
             'your': 57, 'i': 58}
vectorizer = CountVectorizer(vocabulary = word_dict, strip_accents = 'ascii', analyzer = 'word', token_pattern = r"(?u)\b\w+\b")
X_words = vectorizer.fit_transform(df_full['spoken_words'].values.astype('U'))
bag_of_words = vectorizer.get_feature_names()
analyze = vectorizer.build_analyzer()
bag_of_words = [s + ' (spoken)' for s in bag_of_words]
df_words = pd.DataFrame(X_words.toarray(), columns = bag_of_words)
df_1 = pd.concat([df_full, df_words], axis = 1)
#
# Extract Word Data (written in frames)
vectorizer_w = CountVectorizer(vocabulary = word_dict, strip_accents = 'ascii', analyzer = 'word', token_pattern = r"(?u)\b\w+\b")
X_words_w = vectorizer_w.fit_transform(df_full['written_text'].values.astype('U'))
bag_of_words_w = vectorizer_w.get_feature_names()
analyze_w = vectorizer_w.build_analyzer()
bag_of_words_w = [s + ' (written)' for s in bag_of_words_w]
df_words_w = pd.DataFrame(X_words_w.toarray(), columns = bag_of_words_w)
df_2 = pd.concat([df_1, df_words_w], axis = 1)
#
df_2.fillna(0, inplace = True)
#
#
X_full = df_2.drop(columns = ['_id_x', 'content_id', 'Score', '_id_y', 'written_text', 'spoken_words'])
y_full = df_2['Score']
#
###############################################################################
# RANDOM FOREST REGRESSION
###############################################################################
#
# Fitting RFR to the dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
rfr_full = RandomForestRegressor(n_estimators = 1000, criterion = 'mse', max_features = 'auto') #5000
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, y_full, test_size = 0.20)
rfr_full.fit(X_full_train, y_full_train)

# Create predictions and plot outcome of Random Forest model
pred_full_rfr = rfr_full.predict(X_full_test)

rfr_full_feat_imp = pd.DataFrame({'features' : list(X_full_test.columns),
                           'feat_imp' : rfr_full.feature_importances_}, columns=['features','feat_imp'])
rfr_full_feat_imp.to_csv('rfr_feat_imp.csv')

print('\n\nRANDOM FOREST REGRESSION RESULTS')
print('------------------------------------------------------------------')
results_full_rfr = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
#plt.hist2d(y_full_test, pred_full_rfr, bins=50)
#plt.scatter(y_full_test, pred_full_rfr)

plt.scatter(y_full_test[X_full_test['male'] == 1], pred_full_rfr[X_full_test['male'] == 1], c = 'b', s = 2)
plt.scatter(y_full_test[X_full_test['male'] != 1], pred_full_rfr[X_full_test['male'] != 1], c = 'r', s = 2)
plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ls = '--', c = 'k')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('Random Forest Regression')
plt.show()
results_full_rfr.savefig("results_all_RFR.png")

data_rfr = pd.DataFrame({'y_full_test' : y_full_test,
                           'pred_full_rfr' : pred_full_rfr}, columns=['y_full_test','pred_full_rfr'])

g = sns.jointplot("y_full_test", "pred_full_rfr", data = data_rfr, space=0, xlim = (0,10), ylim = (0,10), marginal_kws=dict(bins=20, rug=False), color = 'b', s=5)
g.ax_joint.legend_.remove()
g.set_axis_labels('Actual Score', 'Predicted Score', fontsize=16)
#plt.tightlayout()
plt.show()
g.savefig("RFR_jointplot.png")

r2_rfr_full = r2_score(y_full_test, pred_full_rfr)
print('R2 value: ', r2_rfr_full)

mae_rfr_full = mean_absolute_error(y_full_test, pred_full_rfr)
print('Mean absolute error: ', mae_rfr_full)
#
from sklearn.metrics import mean_squared_error
mse_rfr_full = mean_squared_error(y_full_test, pred_full_rfr)
print('Mean Squared Error: ', mse_rfr_full)
#
###############################################################################
# SUPPORT VECTOR REGRESSION
###############################################################################
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_full_train_sc = sc.fit_transform(X_full_train)
X_full_test_sc = sc.transform(X_full_test)
#
#
from sklearn.svm import SVR
svr_full = SVR(C = 1.0, epsilon = 0.1, kernel = 'rbf')
svr_full.fit(X_full_train_sc, y_full_train)
#
# Create predictions and plot outcome - Support Vector Regression
pred_full_svr = svr_full.predict(X_full_test_sc)
r2_full_svr = r2_score(y_full_test, pred_full_svr)
#
print('\n\nSUPPORT VECTOR REGRESSION RESULTS')
print('------------------------------------------------------------------')
results_full_svr = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.scatter(y_full_test, pred_full_svr)
plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ls = '--', c = 'k')
plt.xlabel('Actual Delta')
plt.ylabel('Predicted Delta')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('Support Vector Regression Results')
plt.show()
results_full_svr.savefig("results_full_SVR.png")
#
print(r2_full_svr)
#
mae_svr_full = mean_absolute_error(y_full_test, pred_full_svr)
print(mae_svr_full)
#
sns.jointplot(y_full_test, pred_full_svr, kind="kde", height=7, space=0)
plt.show()
#
###############################################################################
# KNN REGRESSION
###############################################################################
#
from sklearn.neighbors import KNeighborsRegressor
knn_full = KNeighborsRegressor(n_neighbors=20)
knn_full.fit(X_full_train_sc, y_full_train)
#
# Create predictions and plot outcome - KNN Regression
pred_full_knn = knn_full.predict(X_full_test_sc)
r2_full_knn = r2_score(y_full_test, pred_full_knn)
#
print('\n\nKNN REGRESSION RESULTS')
print('------------------------------------------------------------------')
results_full_knn = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.scatter(y_full_test, pred_full_knn)
plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ls = '--', c = 'k')
plt.xlabel('Actual Delta')
plt.ylabel('Predicted Delta')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('KNN Regression Results')
plt.show()
results_full_knn.savefig("results_full_knn.png")
#
print(r2_full_knn)
#
mae_knn_full = mean_absolute_error(y_full_test, pred_full_knn)
print(mae_knn_full)
#
###############################################################################
# FEATURE SELECTION - RECURSIVE FEATURE ELIMINATION - Linear Model
###############################################################################
#
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select = 70, step=1)
selector = selector.fit(X_full, y_full)
#
import statsmodels.api as sm
X_full_red = sm.add_constant(X_full[X_full.columns[selector.support_]])
X_full_red['humor'] = X_full['humor']
est_red = sm.OLS(y_full.astype(float), X_full_red.astype(float))
est2_red = est_red.fit()
#
# Drop features with high P values and contributing to multicollinearity
X_full_red.drop(['important (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['native'], axis = 1, inplace = True)
X_full_red.drop(['veteran (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['other-mixed'], axis = 1, inplace = True)
X_full_red.drop(['other'], axis = 1, inplace = True)
X_full_red.drop(['working (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['years (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['corporation (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['parenthood (written)'], axis = 1, inplace = True)
X_full_red.drop(['parenthood (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['planned (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['community (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['voted (written)'], axis = 1, inplace = True)
X_full_red.drop(['government (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['we (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['politicians (written)'], axis = 1, inplace = True)
X_full_red.drop(['planned (written)'], axis = 1, inplace = True)
X_full_red.drop(['want (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['change (written)'], axis = 1, inplace = True)
X_full_red.drop(['president (written)'], axis = 1, inplace = True)
X_full_red.drop(X_full_red.columns[10], axis = 1, inplace = True)
X_full_red.drop(['white_male'], axis = 1, inplace = True)
X_full_red.drop(['coal (written)'], axis = 1, inplace = True)
X_full_red.drop(['states (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['narrative'], axis = 1, inplace = True)
X_full_red.drop(['corporation (written)'], axis = 1, inplace = True)
X_full_red.drop(['voted (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['communities (written)'], axis = 1, inplace = True)
X_full_red.drop(['federal (written)'], axis = 1, inplace = True)
X_full_red.drop(['corporations (written)'], axis = 1, inplace = True)
X_full_red.drop(['think (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['tax (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['world (written)'], axis = 1, inplace = True)
X_full_red.drop(['health (spoken)'], axis = 1, inplace = True)
X_full_red.drop(['teaching (written)'], axis = 1, inplace = True)
#
est_red = sm.OLS(y_full.astype(float), X_full_red.astype(float))
est2_red = est_red.fit()
print(est2_red.summary())
#
lin_test_red = np.array(X_full_red).astype(float)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_red = [variance_inflation_factor(lin_test_red, i) for i in range(lin_test_red.shape[1])]
#
output_red = pd.DataFrame({'columns_x' : list(X_full_red.columns.values),
                       'vif_x' : vif_red})
#
X_full_red.drop(['const'], axis = 1, inplace = True)
#
###############################################################################
# LINEAR REGRESSION - reduced predictors
###############################################################################
#
# Create the Linear Regression Training Model using all predictors
from sklearn.linear_model import LinearRegression
lm_full_red = LinearRegression()
X_full_train_red, X_full_test_red, y_full_train_red, y_full_test_red = train_test_split(X_full_red, y_full, test_size = 0.20, random_state = 1)
#
#
# Calculate sample_weights
#weights = y_full_train_opt.apply(lambda x: 1 + (x-5)**2)
#
#
lm_full_red.fit(X_full_train_red,y_full_train_red)
#
# Create predictions and plot outcome
pred_full_lin_red = lm_full_red.predict(X_full_test_red)

print('\n\nLINEAR REGRESSION RESULTS - reduced predictors')
print('------------------------------------------------------------------')
results_full_lm_red = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.scatter(y_full_test_red, pred_full_lin_red, c = 'g', s = 2)
#plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ls = '--', c = 'k')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('Linear Regression')
plt.show()
results_full_lm_red.savefig("results_full_LM_red.png")
#
data_lin_red = pd.DataFrame({'y_full_test_red' : y_full_test_red,
                           'pred_full_lin_red' : pred_full_lin_red}, columns=['y_full_test_red','pred_full_lin_red'])

g1 = sns.jointplot("y_full_test_red", "pred_full_lin_red", kind = 'scatter', data = data_lin_red, space=0, xlim = (0,10), ylim = (0,10), marginal_kws=dict(bins=20, rug=False), color = 'g', s=5)
g1.ax_joint.legend_.remove()
g1.set_axis_labels('Actual Score', 'Predicted Score', fontsize=16)
plt.show()
g1.savefig("LinearReduced_jointplot.png")
#
r2_lin_full_red = r2_score(y_full_test_red, pred_full_lin_red)
print('R2 value: ', r2_lin_full_red)

mae_lin_full_red = mean_absolute_error(y_full_test_red, pred_full_lin_red)
print('Mean absolute error: ', mae_lin_full_red)
#
#
###############################################################################
# POLYNOMIAL REGRESSION - reduced predictors with interaction terms only
###############################################################################
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2, interaction_only = True)
X_full_train_poly = poly.fit_transform(X_full_train_red)
X_full_test_poly = poly.transform(X_full_test_red)
#
poly_full_red = LinearRegression()
poly_full_red.fit(X_full_train_poly,y_full_train_red)

# Create predictions and plot outcome
pred_full_poly_red = poly_full_red.predict(X_full_test_poly)
#
# Create predictions and plot outcome - Support Vector Regression
r2_full_poly = r2_score(y_full_test_red, pred_full_poly_red)

# FIND INTERACTION
data_poly_inter = pd.DataFrame({'coeff' : poly.get_feature_names(),
                           'numbers' : poly_full_red.coef_}, columns=['coeff','numbers'])
#

est_red_poly = sm.OLS(y_full_train.astype(float), X_full_train_poly.astype(float))
est2_red_poly = est_red_poly.fit()
summary_poly = str(est2_red_poly.summary())
f = open('summary_poly.txt','w')
f.write(summary_poly)
f.close()
#
summary_poly_conf = str(est2_red_poly.conf_int(0.05)[0:100]) # 99% confidence interval
f2 = open('summary_poly_conf.txt','w')
f2.write(summary_poly_conf)
f2.close()
#
print('\n\nPOLYNOMIAL REGRESSION RESULTS - interaction reduced predictors')
print('------------------------------------------------------------------')
results_full_poly = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.scatter(y_full_test_red, pred_full_poly_red, c = 'b', s = 1)
plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ls = '--', c = 'k')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('Polynomial Regression Results')
plt.show()
results_full_poly.savefig("results_full_poly.png")
#
print(r2_full_poly)
#
mae_poly_full = mean_absolute_error(y_full_test_red, pred_full_poly_red)
print(mae_poly_full)
#
data_poly_red = pd.DataFrame({'y_full_test_red' : y_full_test_red,
                           'pred_full_poly_red' : pred_full_poly_red}, columns=['y_full_test_red','pred_full_poly_red'])
#
g2 = sns.jointplot("y_full_test_red", "pred_full_poly_red", data = data_poly_red, space=0, xlim = (0,10), ylim = (0,10), marginal_kws=dict(bins=20, rug=False), color = 'r', s=5)
g2.ax_joint.legend_.remove()
g2.set_axis_labels('Actual Score', 'Predicted Score', fontsize=16)
#plt.tightlayout()
plt.show()
g2.savefig("PolyReduced_jointplot.png")
#
print('\n\nLINEAR REGRESSION RESULTS - residuals')
print('------------------------------------------------------------------')
results_full_lm_red_res = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.scatter(pred_full_lin_red,(pred_full_lin_red - y_full_test_red), c = 'b', s = 1)
plt.xlabel('Fitted Values')
plt.ylabel('Rediduals')
plt.xlim(0, 10)
plt.ylim(-5, 5)
plt.title('Linear Regression')
plt.show()
results_full_lm_red_res.savefig("results_full_LM_red_res.png")
#
r2_lin_full_red = r2_score(y_full_test_red, pred_full_lin_red)
print('R2 value: ', r2_lin_full_red)
#
mae_lin_full_red = mean_absolute_error(y_full_test_red, pred_full_lin_red)
print('Mean absolute error: ', mae_lin_full_red)
#
# Plot residuals
residuals_lin = pd.DataFrame({'fitted' : pred_full_lin_red,
 'residuals' : list(pred_full_lin_red - y_full_test_red)}, columns=['fitted','residuals'])
sns.residplot(x = 'fitted', y = 'residuals', data = residuals_lin)
plt.show()
#
#
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(8)
#
plot_lm_1.axes[0] = sns.residplot('fitted', 'residuals', data=residuals_lin, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8,})
#
plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plot_lm_1.savefig("LinearReduced_residuals.png")
#
# Q-Q Curve
import pylab 
import scipy.stats as stats
#
plot_lm_2 = stats.probplot(residuals_lin['residuals'], dist="norm", plot=pylab)
#ax = plot_lm_2.add_subplot(111, figsize=(8, 8))
myplot = pylab.show()
#myplot.savefig("plot_lm_2.png")
#
# Plot residuals - POLY PLOT
residuals_poly = pd.DataFrame({'fitted' : pred_full_poly_red,
 'residuals' : list(pred_full_poly_red - y_full_test_red)}, columns=['fitted','residuals'])
sns.residplot(x = 'fitted', y = 'residuals', data = residuals_poly)
plt.show()
#
#
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(8)
#
plot_lm_1.axes[0] = sns.residplot('fitted', 'residuals', data=residuals_poly, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8,})
#
plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plot_lm_1.savefig("PolyReduced_residuals.png")
#
# Q-Q Curve
import pylab 
import scipy.stats as stats
#
plot_lm_2 = stats.probplot(residuals_poly['residuals'], dist="norm", plot=pylab)
myplot = pylab.show()
#
#
###############################################################################
# PLOT POLYNOMIAL INTERACTIONS
###############################################################################

from numpy.polynomial.polynomial import polyfit
results_interaction = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.scatter(X_full[X_full['white'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['white'] == 1], c = 'b', s = 10)
b, m = polyfit(X_full[X_full['white'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['white'] == 1], 1)
plt.plot(X_full[X_full['white'] == 1]['PoliticalBeliefs'].astype(float), b + m * X_full[X_full['white'] == 1]['PoliticalBeliefs'].astype(float), '-')

plt.scatter(X_full[X_full['white'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['white'] == 0], c = 'r', s = 10)
b1, m1 = polyfit(X_full[X_full['white'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['white'] == 0], 1)
plt.plot(X_full[X_full['white'] == 0]['PoliticalBeliefs'].astype(float), b1 + m1 * X_full[X_full['white'] == 0]['PoliticalBeliefs'].astype(float), '-')
plt.show()
#
#
from numpy.polynomial.polynomial import polyfit
import matplotlib.patches as mpatches
results_interaction = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.scatter(X_full[X_full['black'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['black'] == 1], c = 'r', s = 1)
b, m = polyfit(X_full[X_full['black'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['black'] == 1], 1)
plt.plot(X_full[X_full['black'] == 1]['PoliticalBeliefs'].astype(float), b + m * X_full[X_full['black'] == 1]['PoliticalBeliefs'].astype(float), '-', color = 'r')
plt.scatter(X_full[X_full['black'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['black'] == 0], c = 'b', s = 1)
b1, m1 = polyfit(X_full[X_full['black'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['black'] == 0], 1)
plt.plot(X_full[X_full['black'] == 0]['PoliticalBeliefs'].astype(float), b1 + m1 * X_full[X_full['black'] == 0]['PoliticalBeliefs'].astype(float), '-', color = 'b')
plt.xlabel('Conservative Rating (1-10)')
plt.ylabel('Video Reaction Score')
red_patch = mpatches.Patch(color = 'red', label = 'Ethnicity: black')
blue_patch = mpatches.Patch(color = 'blue', label = 'Ethnicity: non-black')
plt.legend(handles = [red_patch, blue_patch])
plt.show()
results_interaction.savefig("Results_Interaction.png")
#
import statsmodels.api as sm
from statsmodels.formula.api import ols
data = df_2
#
inter_lm_black = ols('Score ~ PoliticalBeliefs + C(black, Sum) + PoliticalBeliefs*C(black, Sum)',
                data=data).fit()
table_black = sm.stats.anova_lm(inter_lm_black, typ=3) # Type 2 Anova DataFrame
print(table_black)

#
# Jitter functions
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, verts=verts, hold=hold, **kwargs)
#
from numpy.polynomial.polynomial import polyfit
import matplotlib.patches as mpatches
results_interaction_trump = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
jitter(X_full[X_full['anti_trump'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['anti_trump'] == 0], c = 'b', s = 0.1)
b1, m1 = polyfit(X_full[X_full['anti_trump'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['anti_trump'] == 0], 1)
plt.plot(X_full[X_full['anti_trump'] == 0]['PoliticalBeliefs'].astype(float), b1 + m1 * X_full[X_full['anti_trump'] == 0]['PoliticalBeliefs'].astype(float), '-', color = 'b')
jitter(X_full[X_full['anti_trump'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['anti_trump'] == 1], c = 'r', s = 0.1)
b, m = polyfit(X_full[X_full['anti_trump'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['anti_trump'] == 1], 1)
plt.plot(X_full[X_full['anti_trump'] == 1]['PoliticalBeliefs'].astype(float), b + m * X_full[X_full['anti_trump'] == 1]['PoliticalBeliefs'].astype(float), '-', color = 'r')
plt.xlabel('Conservative Rating (1-10)')
plt.ylabel('Video Agreement Score')
red_patch = mpatches.Patch(color = 'red', label = 'Anti-Trump Video Message')
blue_patch = mpatches.Patch(color = 'blue', label = 'No Anti-Trump Video Message')
plt.legend(handles = [red_patch, blue_patch])
plt.show()
plt.show()
results_interaction_trump.savefig("Results_Interaction_trump.png")
##############################################################################
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt


def plot_ci_manual(t, s_err, n, x, x2, y2, col, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1]: M. Duarte.  "Curve fitting," JUpyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t*s_err*np.sqrt(1/n + (x2-np.mean(x))**2/np.sum((x-np.mean(x))**2))
    ax.fill_between(x2, y2+ci, y2-ci, color=col, edgecolor="")

    return ax


def plot_ci_bootstrap(n, x, y, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """ 
    if ax is None:
        ax = plt.gca()
    bootindex = sp.random.randint
    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid)-1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(x, y + resamp_resid, 1)                   
        # Plot bootstrap cluster
        ax.plot(x, sp.polyval(pc,x), "b-", linewidth=2, alpha=3.0/float(nboot))

    return ax
# Computations ----------------------------------------------------------------
# Raw Data

x_1 = X_full[X_full['anti_trump'] == 1]['PoliticalBeliefs'].astype(float)
y_1 = y_full[X_full['anti_trump'] == 1]

x_0 = X_full[X_full['anti_trump'] == 0]['PoliticalBeliefs'].astype(float)
y_0 = y_full[X_full['anti_trump'] == 0]

# Modeling with Numpy
p_0, cov_0 = np.polyfit(x_0, y_0, 1, cov=True)        # parameters and covariance from of the fit
y_model_0 = np.polyval(p_0, x_0)                    # model using the fit parameters; NOTE: parameters here are coefficients

p_1, cov_1 = np.polyfit(x_1, y_1, 1, cov=True)        # parameters and covariance from of the fit
y_model_1 = np.polyval(p_1, x_1)                    # model using the fit parameters; NOTE: parameters here are coefficients

# Statistics
n_0 = y_0.size                              # number of observations
m_0 = p_0.size                                  # number of parameters
DF_0 = n_0 - m_0                                    # degrees of freedom
t_0 = stats.t.ppf(0.95, n_0 - m_0)                  # used for CI and PI bands
n_1 = y_1.size                              # number of observations
m_1 = p_1.size                                  # number of parameters
DF_1 = n_1 - m_1                                    # degrees of freedom
t_1 = stats.t.ppf(0.95, n_1 - m_1)                  # used for CI and PI bands

# Estimates of Error in Data/Model
resid_0 = y_0 - y_model_0                           
chi2_0 = np.sum((resid_0/y_model_0)**2)             # chi-squared; estimates error in data
chi2_red_0 = chi2_0/(DF_0)                          # reduced chi-squared; measures goodness of fit
s_err_0 = np.sqrt(np.sum(resid_0**2)/(DF_0))        # standard deviation of the error

resid_1 = y_1 - y_model_1                           
chi2_1 = np.sum((resid_1/y_model_1)**2)             # chi-squared; estimates error in data
chi2_red_1 = chi2_1/(DF_1)                          # reduced chi-squared; measures goodness of fit
s_err_1 = np.sqrt(np.sum(resid_1**2)/(DF_1))        # standard deviation of the error


# Plotting --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,8))
plt.rcParams.update({'font.size': 14})

# Data
jitter(x_0, y_0, c = 'blue', s = 0.2)
jitter(x_1, y_1, c = 'red', s = 0.2)
# Fit
x2 = np.linspace(np.min(x_0), np.max(x_0), 100)
y2 = np.linspace(np.max(y_model_0), np.min(y_model_0), 100)

x3 = np.linspace(np.min(x_1), np.max(x_1), 100)
y3 = np.linspace(np.max(y_model_1), np.min(y_model_1), 100)
#
# Confidence Interval (select one)
plot_ci_manual(t_0, s_err_0, n_0, x_0, x2, y2, 'blue', ax=ax)
plot_ci_manual(t_1, s_err_1, n_1, x_1, x3, y3, 'red', ax=ax)
#plot_ci_bootstrap(n, x, y, resid, ax=ax)

# Prediction Interval
pi_0 = t_0*s_err_0*np.sqrt(1+1/n_0+(x2-np.mean(x_0))**2/np.sum((x_0-np.mean(x_0))**2))   
ax.fill_between(x2, y2+pi_0, y2-pi_0, color="None", linestyle="--")
#ax.plot(x2, y2-pi_0, "--", color="0.5", label="95% Prediction Limits")
#ax.plot(x2, y2+pi_0, "--", color="0.5")

# Prediction Interval
pi_1 = t_1*s_err_1*np.sqrt(1+1/n_1+(x3-np.mean(x_1))**2/np.sum((x_1-np.mean(x_1))**2))   
ax.fill_between(x3, y3+pi_1, y3-pi_1, color="None", linestyle="--")
#ax.plot(x3, y3-pi_1, "--", color="0.5", label="95% Prediction Limits")
#ax.plot(x3, y3+pi_1, "--", color="0.5")

# Figure Modifications --------------------------------------------------------
# Borders
ax.spines["top"].set_color("0.5")
ax.spines["bottom"].set_color("0.5")
ax.spines["left"].set_color("0.5")
ax.spines["right"].set_color("0.5")
ax.get_xaxis().set_tick_params(direction="out")
ax.get_yaxis().set_tick_params(direction="out")
ax.xaxis.tick_bottom()
ax.yaxis.tick_left() 
ax.set_ylim((-0.5, 10.5))
ax.set_xlim((-0.5, 10.5))
#
# Labels
plt.xlabel("Conservative Rating (0-10)", fontsize = '16')
plt.ylabel("Video Agreement Score", fontsize = '16')
plt.tick_params(labelsize = 14)
plt.xlim(np.min(x_0)-1,np.max(x_0)+1)
#
red_patch = mpatches.Patch(color = 'red', label = 'Anti-Trump Message (95% Conf. Lim.)')
blue_patch = mpatches.Patch(color = 'blue', label = 'No Anti-Trump Message (95% Conf. Lim.)')
plt.legend(handles = [red_patch, blue_patch], fontsize = '14')
#
# Custom legend
handles, labels = ax.get_legend_handles_labels()
display = (0,1)
anyArtist = plt.Line2D((0,1),(0,0), color="#b9cfe7")  # Create custom artists
legend = plt.legend(
          [handle for i,handle in enumerate(handles) if i in display]+[anyArtist],
          ["95% Confidence Limits"],
          loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=3, mode="expand")  
frame = legend.get_frame().set_edgecolor("0.5")
#
# Save Figure
plt.tight_layout()
plt.savefig("filename.png")
#
plt.show()
#
import statsmodels.api as sm

x = X_full_red[X_full_red['anti_trump'] == 0]['PoliticalBeliefs'].astype(float)
y = y_full[X_full_red['anti_trump'] == 0].astype(float)
X = sm.add_constant(x)

from statsmodels.sandbox.regression.predstd import wls_prediction_std
re = sm.OLS(y, X).fit()
print(re.summary())
#
prstd, iv_l, iv_u = wls_prediction_std(re)
#
from statsmodels.stats.outliers_influence import summary_table
st, data, ss2 = summary_table(re, alpha=0.05)
#
fittedvalues = data[:, 2]
predict_mean_se  = data[:, 3]
predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
predict_ci_low, predict_ci_upp = data[:, 6:8].T
#
# Check we got the right things
print(np.max(np.abs(re.fittedvalues - fittedvalues)))
print(np.max(np.abs(iv_l - predict_ci_low)))
print(np.max(np.abs(iv_u - predict_ci_upp)))

plt.plot(x, y, 'o')
plt.plot(x, fittedvalues, '-', lw=2)
plt.plot(x, predict_mean_ci_low, 'r--', lw=2)
plt.plot(x, predict_mean_ci_upp, 'r--', lw=2)
plt.show()
#
#######
import statsmodels.api as sm
from statsmodels.formula.api import ols
data = df_2

inter_lm = ols('Score ~ PoliticalBeliefs + C(anti_trump, Sum) + PoliticalBeliefs*C(anti_trump, Sum)',
                data=data).fit()
table = sm.stats.anova_lm(inter_lm, typ=3) # Type 2 Anova DataFrame
print(table)
###############################################################################
#
import seaborn as sns; sns.set(color_codes=True)
data_frame = pd.concat([X_full[['PoliticalBeliefs', 'anti_trump']], y_full], axis = 1).astype(float)
g = sns.lmplot(x='PoliticalBeliefs', y='Score', hue = 'anti_trump', data = data_frame)
plt.show()
#
###############################################################################
from numpy.polynomial.polynomial import polyfit
import matplotlib.patches as mpatches
results_interaction_trump = plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
jitter(X_full[X_full['celebrity'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['celebrity'] == 0], c = 'b', s = 0.1)
b1, m1 = polyfit(X_full[X_full['celebrity'] == 0]['PoliticalBeliefs'].astype(float), y_full[X_full['celebrity'] == 0], 1)
plt.plot(X_full[X_full['celebrity'] == 0]['PoliticalBeliefs'].astype(float), b1 + m1 * X_full[X_full['celebrity'] == 0]['PoliticalBeliefs'].astype(float), '-', color = 'b')
jitter(X_full[X_full['celebrity'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['celebrity'] == 1], c = 'r', s = 0.1)
b, m = polyfit(X_full[X_full['celebrity'] == 1]['PoliticalBeliefs'].astype(float), y_full[X_full['celebrity'] == 1], 1)
plt.plot(X_full[X_full['celebrity'] == 1]['PoliticalBeliefs'].astype(float), b + m * X_full[X_full['celebrity'] == 1]['PoliticalBeliefs'].astype(float), '-', color = 'r')
plt.xlabel('Conservative Rating (1-10)')
plt.ylabel('Video Reaction Score')
red_patch = mpatches.Patch(color = 'red', label = 'Features Celebrity')
blue_patch = mpatches.Patch(color = 'blue', label = 'Does Not Feature Celebrity')
plt.legend(handles = [red_patch, blue_patch])
plt.show()
plt.show()
results_interaction_trump.savefig("Results_Interaction_celebrity.png")

inter_lm_cel = ols('Score ~ PoliticalBeliefs + C(celebrity, Sum) + PoliticalBeliefs*C(celebrity, Sum)',
                data=data).fit()
table_cel = sm.stats.anova_lm(inter_lm_cel, typ=3) # Type 2 Anova DataFrame
print(table_cel)
