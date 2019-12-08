import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import numpy as np

crimes = pd.read_csv('/Users/macbookair/PycharmProjects/Big Data Analytics/Crimes in Boston/crime.csv',
                     encoding='latin-1')
offense_codes = pd.read_csv('/Users/macbookair/PycharmProjects/Big Data Analytics/Crimes in Boston/offense_codes.csv',
                            encoding='latin-1')

# crimes.profile_report().to_file("profilingCrimes.html")

##### Change index and make time series

crimes['OCCURRED_ON_DATE'] = pd.to_datetime(crimes['OCCURRED_ON_DATE'])
crimes = crimes.set_index('OCCURRED_ON_DATE')

# Info on the number of districts
crimes.DISTRICT.unique()
crimes.info()
# Info on null columns
null_columns = crimes.columns[crimes.isnull().any()]
crimes[null_columns].isnull().sum()

# Create df based on type of crimes
crimes_UCR1 = crimes.loc[crimes['UCR_PART'] == 'Part One']
crimes_UCR2 = crimes.loc[crimes['UCR_PART'] == 'Part Two']
crimes_UCR3 = crimes.loc[crimes['UCR_PART'] == 'Part Three']

# Check null values in each column
null_columns = crimes_UCR1.columns[crimes.isnull().any()]
crimes_UCR1[null_columns].isnull().sum()

null_columns = crimes_UCR2.columns[crimes.isnull().any()]
crimes_UCR2[null_columns].isnull().sum()

null_columns = crimes_UCR2.columns[crimes.isnull().any()]
crimes_UCR3[null_columns].isnull().sum()

#### Plot crimes to look for patterns
nb_crimes_UCR1 = crimes_UCR1.index.value_counts().resample('D').sum()
nb_crimes_UCR1.plot()
plt.title('UCR Type 1 Crimes')
plt.show()

nb_crimes_UCR2 = crimes_UCR2.index.value_counts().resample('D').sum()
nb_crimes_UCR2.plot()
plt.title('UCR Type 2 Crimes')
plt.show()

nb_crimes_UCR3 = crimes_UCR3.index.value_counts().resample('D').sum()
nb_crimes_UCR3.plot()
plt.title('UCR Type 3 Crimes')
plt.show()

plt.plot(nb_crimes_UCR1.index, nb_crimes_UCR1, label='UCR1')
plt.plot(nb_crimes_UCR2.index, nb_crimes_UCR2, label='UCR2')
plt.plot(nb_crimes_UCR3.index, nb_crimes_UCR3, label='UCR3')
plt.legend()
plt.show()

######### Let's explore the datasets a bit more

### Dataset start and end dates
print('Start: {} - End: {}'.format(min(crimes.index), max(crimes.index)))

### Check for aggregated weekday patterns

# Aggregated through the whole time line
UCR1_days = crimes_UCR1.groupby(crimes_UCR1.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR2_days = crimes_UCR2.groupby(crimes_UCR2.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR3_days = crimes_UCR3.groupby(crimes_UCR3.DAY_OF_WEEK).count()['INCIDENT_NUMBER']

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

plt.plot(day_order, UCR1_days, label='UCR1')
plt.plot(day_order, UCR2_days, label='UCR2')
plt.plot(day_order, UCR3_days, label='UCR3')
plt.title('Aggregated crimes by day')
plt.legend()
plt.show()

# Isolate type 1 because it gets crushed
plt.plot(day_order, UCR1_days, label='UCR1')
plt.title('Aggregated UCR1 crimes by day')
plt.legend()
plt.show()

### Split data by year
UCR1_201516 = crimes_UCR1.loc['2015-06-15':'2016-06-14']
UCR1_201617 = crimes_UCR1.loc['2016-06-15':'2017-06-14']
UCR1_201718 = crimes_UCR1.loc['2017-06-15':'2018-06-14']
UCR1_201819 = crimes_UCR1.loc['2018-06-15':]

UCR2_201516 = crimes_UCR2.loc['2015-06-15':'2016-06-14']
UCR2_201617 = crimes_UCR2.loc['2016-06-15':'2017-06-14']
UCR2_201718 = crimes_UCR2.loc['2017-06-15':'2018-06-14']
UCR2_201819 = crimes_UCR2.loc['2018-06-15':]

UCR3_201516 = crimes_UCR3.loc['2015-06-15':'2016-06-14']
UCR3_201617 = crimes_UCR3.loc['2016-06-15':'2017-06-14']
UCR3_201718 = crimes_UCR3.loc['2017-06-15':'2018-06-14']
UCR3_201819 = crimes_UCR3.loc['2018-06-15':]

### Groups weekdays by year
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# For UCR1
UCR1_days_201516 = UCR1_201516.groupby(UCR1_201516.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR1_days_201617 = UCR1_201617.groupby(UCR1_201617.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR1_days_201718 = UCR1_201718.groupby(UCR1_201718.DAY_OF_WEEK).count()['INCIDENT_NUMBER']

# For UCR2
UCR2_days_201516 = UCR2_201516.groupby(UCR2_201516.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR2_days_201617 = UCR2_201617.groupby(UCR2_201617.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR2_days_201718 = UCR2_201718.groupby(UCR2_201718.DAY_OF_WEEK).count()['INCIDENT_NUMBER']

# For UCR3
UCR3_days_201516 = UCR3_201516.groupby(UCR3_201516.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR3_days_201617 = UCR3_201617.groupby(UCR3_201617.DAY_OF_WEEK).count()['INCIDENT_NUMBER']
UCR3_days_201718 = UCR3_201718.groupby(UCR3_201718.DAY_OF_WEEK).count()['INCIDENT_NUMBER']

# Make a big plot with everything
plt.plot(day_order, UCR1_days_201516, label='UCR1 2015-16', color='red', linestyle='-')
plt.plot(day_order, UCR1_days_201617, label='UCR1 2016-17', color='red', linestyle='--')
plt.plot(day_order, UCR1_days_201718, label='UCR1 2017-18', color='red', linestyle=':')
plt.plot(day_order, UCR2_days_201516, label='UCR2 2015-16', color='blue', linestyle='-')
plt.plot(day_order, UCR2_days_201617, label='UCR2 2016-17', color='blue', linestyle='--')
plt.plot(day_order, UCR2_days_201718, label='UCR2 2017-18', color='blue', linestyle=':')
plt.plot(day_order, UCR3_days_201516, label='UCR3 2015-16', color='green', linestyle='-')
plt.plot(day_order, UCR3_days_201617, label='UCR3 2016-17', color='green', linestyle='--')
plt.plot(day_order, UCR3_days_201718, label='UCR3 2017-18', color='green', linestyle=':')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.title('Number of crimes per day')
plt.show()

### Groups months by year

# For UCR1
UCR1_months_201516 = UCR1_201516.groupby(UCR1_201516.MONTH).count()['INCIDENT_NUMBER'].to_frame()
UCR1_months_201617 = UCR1_201617.groupby(UCR1_201617.MONTH).count()['INCIDENT_NUMBER'].to_frame()
UCR1_months_201718 = UCR1_201718.groupby(UCR1_201718.MONTH).count()['INCIDENT_NUMBER'].to_frame()

# For UCR2
UCR2_months_201516 = UCR2_201516.groupby(UCR2_201516.MONTH).count()['INCIDENT_NUMBER'].to_frame()
UCR2_months_201617 = UCR2_201617.groupby(UCR2_201617.MONTH).count()['INCIDENT_NUMBER'].to_frame()
UCR2_months_201718 = UCR2_201718.groupby(UCR2_201718.MONTH).count()['INCIDENT_NUMBER'].to_frame()

# For UCR3
UCR3_months_201516 = UCR3_201516.groupby(UCR3_201516.MONTH).count()['INCIDENT_NUMBER'].to_frame()
UCR3_months_201617 = UCR3_201617.groupby(UCR3_201617.MONTH).count()['INCIDENT_NUMBER'].to_frame()
UCR3_months_201718 = UCR3_201718.groupby(UCR3_201718.MONTH).count()['INCIDENT_NUMBER'].to_frame()

# Make a big plot with everything
plt.plot(UCR1_months_201516.index, UCR1_months_201516, label='UCR1 2015-16', color='red', linestyle='-')
plt.plot(UCR1_months_201516.index, UCR1_months_201617, label='UCR1 2016-17', color='red', linestyle='--')
plt.plot(UCR1_months_201516.index, UCR1_months_201718, label='UCR1 2017-18', color='red', linestyle=':')
plt.plot(UCR1_months_201516.index, UCR2_months_201516, label='UCR2 2015-16', color='blue', linestyle='-')
plt.plot(UCR1_months_201516.index, UCR2_months_201617, label='UCR2 2016-17', color='blue', linestyle='--')
plt.plot(UCR1_months_201516.index, UCR2_months_201718, label='UCR2 2017-18', color='blue', linestyle=':')
plt.plot(UCR1_months_201516.index, UCR3_months_201516, label='UCR3 2015-16', color='green', linestyle='-')
plt.plot(UCR1_months_201516.index, UCR3_months_201617, label='UCR3 2016-17', color='green', linestyle='--')
plt.plot(UCR1_months_201516.index, UCR3_months_201718, label='UCR3 2017-18', color='green', linestyle=':')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.xticks(range(1,13))
plt.title('Number of crimes per month')
plt.show()