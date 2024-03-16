import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('air1.csv', encoding='unicode_escape', on_bad_lines='skip')

df.isnull().sum()

df['state'].value_counts()

df['type'].value_counts()

df['agency'].value_counts()

nullvalues = df.isnull().sum().sort_values(ascending=False)

df.fillna(0, inplace=True)
df.isnull().sum()


def cal_soi(so2):
    S=float(so2)
    si = 0
    if (S<= 40):
        si = S * (50 / 40)
    elif (S > 40 and S <= 80):
        si = 50 + (S - 40) * (50 / 40)
    elif (S> 80 and S<= 380):
        si = 100 + (S - 80) * (100 / 300)
    elif (S> 380 and so2 <= 800):
        si = 200 + (S - 380) * (100 / 420)
    elif (S > 800 and S<= 1600):
        si = 300 + (S- 800) * (100 / 800)
    elif (S > 1600):
        si = 400 + (S - 1600) * (100 / 800)
    return si


df['Soi'] = df['so2'].apply(cal_soi)
data = df[['so2', 'Soi']]


def cal_Noi(no2):
    ni = 0
    if (no2 <= 40):
        ni = no2 * 50 / 40
    elif (no2 > 40 and no2 <= 80):
        ni = 50 + (no2 - 40) * (50 / 40)
    elif (no2 > 80 and no2 <= 180):
        ni = 100 + (no2 - 80) * (100 / 100)
    elif (no2 > 180 and no2 <= 280):
        ni = 200 + (no2 - 180) * (100 / 100)
    elif (no2 > 280 and no2 <= 400):
        ni = 300 + (no2 - 280) * (100 / 120)
    else:
        ni = 400 + (no2 - 400) * (100 / 120)
    return ni


df['Noi'] = df['no2'].apply(cal_Noi)
data = df[['no2', 'Noi']]



def cal_RSPMI(rspm):
    rpi = 0
    if (rpi <= 30):
        rpi = rpi * 50 / 30
    elif (rpi > 30 and rpi <= 60):
        rpi = 50 + (rpi - 30) * (50 / 30)
    elif (rpi > 60 and rpi <= 90):
        rpi = 100 + (rpi - 60) * (100 / 30)
    elif (rpi > 90 and rpi <= 120):
        rpi = 200 + (rpi - 90) * (100 / 30)
    elif (rpi > 120 and rpi <= 250):
        rpi = 300 + (rpi - 120) * (100 / 130)
    else:
        rpi = 400 + (rpi - 250) * (100 / 130)
    return rpi


df['Rpi'] = df['rspm'].apply(cal_RSPMI)
data = df[['rspm', 'Rpi']]


def cal_SPMI(spm):
    spi = 0
    if (spm <= 50):
        spi = spm * 50 / 30
    elif (spm > 50 and spm <= 100):
        spi = 50 + (spm - 50) * (50 / 50)
    elif (spm > 100 and spm <= 250):
        spi = 100 + (spm - 100) * (100 / 150)
    elif (spm > 250 and spm <= 350):
        spi = 200 + (spm - 250) * (100 / 100)
    elif (spm > 350 and spm <= 430):
        spi = 300 + (spm - 350) * (100 / 80)
    else:
        spi = 400 + (spm - 430) * (100 / 430)
    return spi


df['SPMi'] = df['spm'].apply(cal_SPMI)
data = df[['spm', 'SPMi']]


def cal_aql(si, ni, rpi, spi):
    aqi = 0
    if (si > ni and si > rpi and si > spi):
        aqi = si
    if (ni > si and ni > rpi and ni > spi):
        aqi = ni
    if (rpi > si and rpi > ni and rpi > spi):
        aqi = rpi
    if (spi > si and spi > ni and spi > rpi):
        aqi = spi
    return aqi


df['AQI'] = df.apply(lambda x: cal_aql(x['Soi'], x['Noi'], x['Rpi'], x['SPMi']), axis=1)
data = df[['state', 'Soi', 'Noi', 'Rpi', 'SPMi', 'AQI']]


def AQI_Range(x):
    if x <= 50:
        return "Good"
    elif x > 50 and x <= 100:
        return "Moderate"
    elif x > 100 and x <= 200:
        return "Poor"
    elif x > 200 and x <= 300:
        return "Unhealthy"
    elif x > 300 and x <= 400:
        return "Very unhealthy"
    elif x > 400:
        return "Hazardous"


df['AQI_Range'] = df['AQI'].apply(AQI_Range)

X = df[['Soi', 'Noi', 'Rpi', 'SPMi']]
Y = df['AQI']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=70)

RF = RandomForestRegressor().fit(X_train, Y_train)

pickle.dump(RF, open("air.clf", 'wb'))

