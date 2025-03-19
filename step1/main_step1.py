import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
if __name__ == '__main__':
    st.title("Анализ погоды с использованием Streamlit")
    st.header("Шаг 1")

    uploaded_file = st.file_uploader("temperature_data.csv", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['date_time'] = pd.to_datetime(data['timestamp'])
        data.drop(['timestamp'], axis=1, inplace=True)
    #    проверяла данные
        st.write("Превью данных:")
        st.dataframe(data.head())
        # вставила мини предобработку данных
        # if st.checkbox('Do you want to delete miss position?'):
        #     data = data.dropna()
        #     st.write("missed position was deleted")
        # if st.checkbox('Do you want to change missed position on mean??'):
        #     data = data.fillna(data.mean())
        #     st.write("missed position was changed")
        city = st.selectbox("Select city", data['city'].unique())
        print(city)
        if city:
            selected_city = data[data['city'] == city]
            selected_city['mv_mean'] = selected_city['temperature'].rolling(window=30).mean()
            st.write("grafik")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(selected_city['date_time'], selected_city['temperature'], label='Temperature', alpha=0.5)
            ax.plot(selected_city['date_time'], selected_city['mv_mean'], label='mean_30days', color='red')
            ax.set_xlabel("date_time")
            ax.set_ylabel("temperature")
            ax.set_title(f"Temperature in {city} c 30days mean")
            ax.legend()
            st.pyplot(fig)

            # анализ ряда
            # result = adfuller(selected_city['temperature'])
            # st.write("Проверяем стационарность")
            # st.write('ADF Statistic: %f' % result[0])
            # st.write('p-value: %f' % result[1])
            # st.write('Critical Values:')
            # for key, value in result[4].items():
            #     st.write('\t%s: %.3f' % (key, value))
            # if result[1] < 0.05:
            #     st.write("Считаем ряд стационарным")
            # else:
            #     st.write("Считаем ряд нестационарным и преобразовываем данные")
            #     selected_city['no_trend_temp'] = selected_city['temperature'] - selected_city['temperature'].rolling(window=30).mean()
            #
            #     # Преобразование для удаления сезонности (в данном случае просто разница между текущим и предыдущим значением)
            #     selected_city['station_temp'] = selected_city['no_trend_temp'].diff()
            #
            #     # Построим графики
            #     fig2, ax2 = plt.subplots(figsize=(10, 5))
            #     ax2.plot(selected_city['date_time'], selected_city['no_trend_temp'], label='Без тренда')
            #     ax2.plot(selected_city['date_time'], selected_city['station_temp'], label='Стационарные')
            #     ax2.legend()
            #     ax2.title('Преобразованные данные')
            #     ax2.xlabel('Дата')
            #     ax2.ylabel('temperature')
            #     st.pyplot(fig2)
            selected_city.set_index(['date_time'], inplace=True)
            decomp = sm.tsa.seasonal_decompose(selected_city['temperature'], model='additive', period=365)
            fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
            decomp.observed.plot(ax=ax1, title="Исходный ряд")
            decomp.trend.plot(ax=ax2, title="Тренд")
            decomp.seasonal.plot(ax=ax3, title="Сезонность")
            decomp.resid.plot(ax=ax4, title="Остатки")
            st.pyplot(fig2)

            st.write("### Прогнозирование температуры")
            model = ARIMA(selected_city["temperature"], order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=30)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(selected_city.index, selected_city["temperature"], label="Фактические данные")
            ax.plot(pd.date_range(start=selected_city.index[-1], periods=31, freq='D')[1:], forecast, label="Прогноз",
                    color='green')
            ax.set_xlabel("Дата")
            ax.set_ylabel("Температура")
            ax.set_title(f"Прогноз температуры в {city} на 30 дней")
            ax.legend()
            st.pyplot(fig)

            # Поиск аномалий
            st.write("### Поиск аномалий в температуре")
            selected_city["z_score"] = (selected_city["temperature"] - selected_city["temperature"].mean()) / selected_city["temperature"].std()
            anomalies = selected_city[np.abs(selected_city["z_score"]) > 3]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(selected_city.index, selected_city["temperature"], label="Температура")
            ax.scatter(anomalies.index, anomalies["temperature"], color='red', label="Аномалии")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Температура")
            ax.set_title(f"Аномалии температуры в {city}")
            ax.legend()
            st.pyplot(fig)

            if not anomalies.empty:
                st.write("Обнаруженные аномалии:")
                st.write(anomalies)


    else:
        st.write("Пожалуйста, загрузите CSV-файл.")