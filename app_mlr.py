import streamlit as slt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Page Config #
slt.set_page_config('Multiple Linear Regression', layout='centered')


# Load CSS #

def load_css(file):
    with open (file) as f:
        slt.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
load_css('style.css')

slt.markdown('''
    <div class="card">
        <h1>Multiple Linear Regression</h1>
        <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Multiple Linear Regression</p>
    </div>
''', unsafe_allow_html=True)

# Load Data

@slt.cache_data

def load_data():
    return sns.load_dataset('tips')

df = load_data()

# Dataset Preview #

slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader('Dataset Preview')
slt.dataframe(df[["total_bill", "size", "tip"]].head())
slt.markdown('</div>', unsafe_allow_html=True)


# Prepare data #

x, y = df[["total_bill", "size"]], df["tip"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train = pd.DataFrame(
    scaler.fit_transform(x_train),
    columns=[["total_bill", "size"]]
)

x_test = pd.DataFrame(
    scaler.transform(x_test),
    columns=[["total_bill", "size"]]
)


# Train model #

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# Metrics #

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)


# Visualization #

slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader('Total Bill vs Tip (with Multiple Lineaar Regression)')
fig, ax = plt.subplots()
scatter = ax.scatter(df["total_bill"], df["tip"], c=df["size"], cmap='coolwarm', alpha=0.6)
plt.colorbar(scatter, label='size')
y_line = model.predict(scaler.transform(x_test))

# ax.plot(
#     df.loc[x_test.index, "total_bill"],
#     y_line,
#     color="red"
# )
ax.set_ylabel("Tip Amount ($)")
slt.pyplot(fig)

slt.markdown('</div>', unsafe_allow_html=True)


# m & c #

slt.markdown(f'''
    <div class="card">
        <h3>Model Interception</h3>
        <p><b>Co-efficient(Total bill): </b> {model.coef_[0]: .3f}<p>
        <p><b>Co-efficient(Group Size): </b> {model.coef_[1]: .3f}</p>
        <p><b>Intercept: </b> {model.intercept_: .3f}</p>
        <p>
            Tip depends up on the <b> Bill Amount </b> and <b> Group size </b>
        </p>
    </div>             
''', unsafe_allow_html=True)


# Prediction #

slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader('Predict Tip Amount')

bill = slt.slider(
    "Total Bill ($)",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    30.0
)

size = slt.slider(
    "Group Size",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)

input_df = pd.DataFrame(
    [[bill, size]],
    columns=["total_bill", "size"]
)

input_scaled = scaler.transform(input_df)
tip = model.predict(input_scaled)[0]   # ✅ extract scalar

slt.markdown(
    f'<div class="prediction-box"> Predicted Tip : ($) {tip:.2f} </div>',
    unsafe_allow_html=True
)

slt.markdown('</div>', unsafe_allow_html=True)
