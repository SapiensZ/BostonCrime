import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling
import fbprophet

# Read data
df = pd.read_csv('crime.csv', encoding='iso-8859-1')
offense_codes = pd.read_csv('offense_codes.csv', encoding='iso-8859-1')

# Set DateTime Index
df.columns = [str.lower(col) for col in df.columns]
df['occurred_on_date'] = pd.to_datetime(df.occurred_on_date)
df = df.set_index('occurred_on_date')
print('Data first entry: {}, last entry: {}'.format(df.index.min(), df.index.max()))

# Group by day
daily_crimes = pd.DataFrame(df[df.district=='B2'].groupby(pd.Grouper(freq='D')).count()['incident_number'])
daily_crimes['ds'] = daily_crimes.index
daily_crimes.columns = ['y', 'ds']
reporting_start_date, reporting_end_date = df.index.min(), df.index.max()

# Facebook Prophet forecasting
m = fbprophet.Prophet()
m.fit(daily_crimes)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.title("Forecast")
plt.show()

fig2 = m.plot_components(forecast)
plt.title("Components")
plt.show()


def plot_crimes(frame, start_date=reporting_start_date, end_date=reporting_end_date):
    frame.loc[start_date:end_date].plot()
    plt.title('Crimes from {} to {} for {}'.format(start_date, end_date, part))
    plt.show()

for part in ['Part One', 'Part Two', 'Part Three']:
    part_crimes = df[df.ucr_part == part].groupby(pd.Grouper(freq='D')).count()['incident_number']
    plot_crimes(part_crimes, daily_crimes.loc['2016-08-15': '2016-09-15'])