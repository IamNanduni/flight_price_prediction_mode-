# %%
pip install plotly

# %%
pip install statsmodels

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from plotly.offline import init_notebook_mode, iplot

# Initialize Plotly notebook mode
init_notebook_mode(connected=True)

# %%
# Load dataset
df = pd.read_csv("Clean_Dataset.csv")

# Display basic info about the dataset
print(df.head())
print(df.info())
print(df.describe().T[['min', 'mean', 'std', '50%', 'max']].style.background_gradient(axis=1))

# Drop Unnamed column
df.drop(columns=df.columns[0], inplace=True)
print(df.head())

# %%
# EDA: Plot most frequent airlines
cati = np.array(df['airline'].value_counts())
labels = df['airline'].value_counts().keys()
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=11
color_palette_list = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9', '#C1F0F6', '#0099CC']
plt.pie(cati, labels=labels, startangle=90, colors=color_palette_list, autopct='%1.0f%%', explode=(0,0,0,0.1,0.2,0.3))
plt.show()

# %%
!pip install --upgrade nbformat

# %%
import nbformat
print(nbformat.__version__)

# %%
# Avg pricing for airlines
mean_airlines = df.groupby('airline')['price'].mean()
fig = px.histogram(data_frame=df, x=mean_airlines.index, y=mean_airlines.values, title='Average Pricing Trip for Airlines')
fig.show()

# Class distribution in airlines
fig = px.histogram(data_frame=df, y='airline', color='class', title='Class in Airlines')
fig.show()

# Avg pricing for business and economy class
fig = px.histogram(data_frame=df, x='airline', y='price', color='class', title='Avg Trip Pricing for Business and Economy', histfunc="avg", barmode='group', text_auto=True)
fig.show()

# Business data exploration
business_data = df[df['class'] == 'Business']
print(business_data['source_city'].value_counts())
print(business_data['destination_city'].value_counts())

# From Mumbai to destination cities
mumbai_source = business_data[business_data['source_city']=='Mumbai']
mu_labels = mumbai_source['destination_city'].value_counts().keys()
mu_values = mumbai_source['destination_city'].value_counts().values
fig = go.Figure(data=[go.Pie(labels=mu_labels, values=mu_values, textinfo='label+percent', insidetextorientation='radial', hole=0.35)])
fig.update_layout(title_text="From Mumbai to Destination Cities")
fig.show()

# From Delhi to destination cities
delhi_source = business_data[business_data['source_city']=='Delhi']
d_labels = delhi_source['destination_city'].value_counts().keys()
d_values = delhi_source['destination_city'].value_counts().values
fig = px.pie(delhi_source, values=d_values, names=d_labels, color=d_labels, hole=0.3, title="From Delhi to Destination Cities", color_discrete_map={'Mumbai':'lightcyan', 'Bangalore':'cyan', 'Kolkata':'royalblue', 'Hyderabad':'darkblue', 'Chennai':'lightblue'})
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

# Stops per business travels
sns.countplot(data=mumbai_source, x='destination_city', hue='stops')
plt.title('Stops per Business Travels')
plt.show()

# Departure time distribution in business travels
print(mumbai_source['departure_time'].value_counts())

# Filter by departure time
mumbai_source_deptime = mumbai_source[(mumbai_source['departure_time'] == 'Early_Morning') | (mumbai_source['departure_time'] == 'Evening') | (mumbai_source['departure_time'] == 'Morning')]
fig = px.histogram(data_frame=mumbai_source_deptime, x='departure_time', color='arrival_time', barmode='group', text_auto=True, title='Departure Time & Arrival Time')
fig.show()

# Economy data exploration
eco_data = df[df['class'] == 'Economy']

# Relation between days left and price
sns.lineplot(data=eco_data, x='days_left', y='price')
plt.xlim(0, 40)
plt.show()

# Arrival time & departure time in economy class
fig = px.histogram(data_frame=eco_data, x='departure_time', color='arrival_time', barmode='group', text_auto=True, title='Arrival Time & Departure Time')
fig.show()

# %%
# Modeling: Drop flight column
data_modeling = df.drop(columns=['flight'])
print(data_modeling.head())

# Encoding categorical data
le = LabelEncoder()
data_modeling_encoded = data_modeling.copy()
columns_to_encoding = data_modeling.columns[:7]
for column in columns_to_encoding:
    data_modeling_encoded[column] = le.fit_transform(data_modeling_encoded[column])
print(data_modeling_encoded.head())

# Split X (features) and y (target)
X = data_modeling_encoded.iloc[:, :-1]
y = data_modeling_encoded.iloc[:, -1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

y_predict = lr.predict(X_test)
residuals = np.absolute(y_predict - y_test)
print("Root Mean Square Error:", mean_squared_error(y_true=y_test, y_pred=y_predict, squared=False))

# Modeling using statsmodels
X_train_stat = sm.add_constant(X_train)
X_test_stat = sm.add_constant(X_test)
model = sm.OLS(y_train, X_train_stat)
results_of_firstModel = model.fit()
print(results_of_firstModel.summary())

# Check normality assumption for regression model
fig = sm.qqplot(results_of_firstModel.resid, fit=True, line='45')
plt.show()

# KDE plot of residuals
sns.kdeplot(results_of_firstModel.resid, fill=True)
plt.show()

# Check equal variance assumption for regression model
sns.scatterplot(x=results_of_firstModel.predict(X_train_stat), y=results_of_firstModel.resid)
plt.show()

# New model with log of y
log_y = np.log(y_train)
new_X_train = sm.add_constant(X_train)
new_X_test = sm.add_constant(X_test)
model2 = sm.OLS(log_y, new_X_train)
results_of_Second_Model = model2.fit()
print(results_of_Second_Model.summary())

# Check normality assumption for new model with log y
fig = sm.qqplot(results_of_Second_Model.resid, fit=True, line='45')
plt.show()

# KDE plot of residuals for new model with log y
sns.kdeplot(results_of_Second_Model.resid, fill=True)
plt.show()

# Check equal variance assumption for new model with log y
sns.scatterplot(x=results_of_Second_Model.predict(new_X_train), y=results_of_Second_Model.resid)
plt.show()



