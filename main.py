#modules
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go








st.title('Stock Prediction App')




#app
start_date = "2015-01-01"
today_date = date.today().strftime("%Y-%m-%d")



stocks = ("AAPL", "MSFT", "GOOG", "AMZN", "TSLA","TSM") #diffrent stock tickers


user_input = st.text_input("Enter Stock ticker Symbol, Ex: AAPL, MSFT, GOOG ......")

#stock_selection = st.selectbox("Selected dataset for predction", stocks)

num_years = st.slider("Years of prediction",1,4)

period = num_years * 365

while user_input != "": #don't do visualization/data crunching until user enters stock ticker
    @st.cache #cache the data for faster loading
    def get_data(user_input):
        data = yf.download(user_input, start_date, today_date) #calling yfinance to get data
        data.reset_index(inplace = True)
        return data


    data_state = st.text("Loading data...")
    data = get_data(user_input)
    data_state.text("Data loaded!")

    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_chart_data():
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        chart.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        chart.layout.update(title_text='Chart Data with a slider', xaxis_rangeslider_visible=True)
        st.plotly_chart(chart)
        
    plot_chart_data()


    #forecasting to predicition model
    stock_train = data[['Date', 'Close']]
    stock_train = stock_train.rename(columns={'Date': 'ds', 'Close': 'y'}) #passing to fbprohet

    training_model = Prophet() #new training model instance

    training_model.fit(stock_train) #starts the training

    future_forecast = training_model.make_future_dataframe(periods=period) #creates a future dataframe

    future_data = training_model.predict(future_forecast) #making predictions


    st.subheader('Prediction Model')
    st.write(future_data.tail()) #last 10 rows


    dataf1 = plot_plotly(training_model, future_data) #plotting the predictions, assing a df for a plot
    st.plotly_chart(dataf1) #plotting to app

    st.write('forecast componenets')
    dataf2 = training_model.plot_components(future_data) #plotting the components of the predictions
    st.write(dataf2)
