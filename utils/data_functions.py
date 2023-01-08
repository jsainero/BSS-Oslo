from functools import reduce
import os
import pandas as pd
pd.options.mode.chained_assignment = None


def count_trips(df, direction):

    if direction not in ('inbound', 'outbound'):
        raise Exception('Invalid value for direction')
    else:
        trip_time = 'started_at' if direction == 'outbound' else 'ended_at'
        trip_station = 'start_station_id' if direction == 'outbound' else 'end_station_id'
        df_filtered = df[[trip_time, trip_station, 'duration']]
        df_filtered[trip_time] = pd.to_datetime(df_filtered[trip_time])
        df_filtered['day'] = df_filtered[trip_time].dt.date
        df_filtered['hour'] = df_filtered[trip_time].dt.hour
        df_filtered['day_hour'] = pd.to_datetime(
            df_filtered['day']) + pd.to_timedelta(df_filtered['hour'], unit='h')
        df_filtered = df_filtered[['day_hour', trip_station, 'duration']]
        df_filtered.columns = ['day_hour', 'station_id', 'duration']
        df_filtered['trips'] = 1.0
        df_grouped = df_filtered.groupby(['day_hour', 'station_id'], as_index=False).agg({
            'duration': 'mean', 'trips': 'sum'})

    return df_grouped


def stations_data(df):

    column_names = ['id', 'name', 'latitude', 'longitude']
    df_filtered_outbound = df[['start_station_id', 'start_station_name',
                               'start_station_latitude', 'start_station_longitude']]
    df_filtered_inbound = df[['end_station_id', 'end_station_name',
                              'end_station_latitude', 'end_station_longitude']]
    df_filtered_outbound.columns = column_names
    df_filtered_inbound.columns = column_names
    df_filtered = df_filtered_outbound.append(df_filtered_inbound)

    return df_filtered.drop_duplicates()


def weather_phenomema_agg(x, y):
    if x == 'snow' or y == 'snow':
        return 'snow'
    elif x == 'rain' or y == 'rain':
        return 'rain'
    elif x == 'fog' or y == 'fog':
        return 'fog'
    else:
        return 'no phenomena'


def weather_phenomema_series(series):
    return reduce(lambda x, y: weather_phenomema_agg(x, y), series)


def weather_data(filepath):

    df = pd.read_csv(filepath, sep=';', comment='#',
                     encoding='cp1252', index_col=False)

    df_cleaned = df.iloc[:, [0, 1, 7, 12]]
    df_cleaned.columns = ['time', 'temperature',
                          'wind_speed', 'weather_phenomena']

    df_cleaned['weather_phenomena'] = df_cleaned['weather_phenomena'].fillna(
        'no phenomena')
    df_cleaned['weather_phenomena'] = df_cleaned['weather_phenomena'].str.lower()
    df_cleaned.loc[df_cleaned['weather_phenomena'].str.contains(
        'snow'), 'weather_phenomena'] = 'snow'
    df_cleaned.loc[df_cleaned['weather_phenomena'].str.contains(
        'rain|drizzle|thunderstorm|shower'), 'weather_phenomena'] = 'rain'
    df_cleaned.loc[df_cleaned['weather_phenomena'].str.contains(
        'fog|mist|cloud'), 'weather_phenomena'] = 'fog'

    df_cleaned['time'] = pd.to_datetime(
        df_cleaned['time'], format='%d.%m.%Y %H:%M')
    # df_cleaned = df_cleaned.set_index('time')
    # df_cleaned.index = df_cleaned.index.tz_localize('Europe/Oslo', ambiguous='infer', nonexistent='shift_forward').tz_convert(pytz.utc).tz_convert(None)
    # df_cleaned = df_cleaned.reset_index()
    df_cleaned.time = df_cleaned.time.dt.round('H')

    agg_dict = {
        'temperature': 'mean',
        'wind_speed': 'mean',
        'weather_phenomena': weather_phenomema_series
    }

    df_grouped = df_cleaned.groupby(['time'], as_index=False).agg(agg_dict)
    df_grouped = df_grouped.set_index('time')

    return df_grouped


def read_monthly_data(filepath):

    df = pd.read_csv(filepath)
    df_outbounds = count_trips(df, 'outbound')
    df_inbounds = count_trips(df, 'inbound')
    df_stations = stations_data(df)

    return df_outbounds, df_inbounds, df_stations


def read_data(dirpath):

    df_outbounds = pd.DataFrame(
        columns=['day_hour', 'station_id', 'duration', 'trips'])
    df_inbounds = pd.DataFrame(
        columns=['day_hour', 'station_id', 'duration', 'trips'])
    df_stations = pd.DataFrame(columns=['id', 'name', 'latitude', 'longitude'])

    for year in os.scandir(dirpath+'/bike_trips'):
        for month in os.scandir(year):
            monthly_outbounds, monthly_inbounds, monthly_stations = read_monthly_data(
                month)
            df_outbounds = df_outbounds.append(monthly_outbounds)
            df_inbounds = df_inbounds.append(monthly_inbounds)
            df_stations = df_stations.append(monthly_stations)

    df_weather = weather_data(
        dirpath+'/weather/oslo_weather_20200101_20221101.csv')

    return df_outbounds, df_inbounds, df_stations.drop_duplicates().reset_index(drop=True), df_weather


def join_dfs_for_analysis(outbounds_df, inbounds_df, weather_df, start_date=None, end_date=None):
    outbounds = outbounds_df
    inbounds = inbounds_df

    if start_date:
        outbounds = outbounds[outbounds.day_hour >= start_date]
        inbounds = inbounds[inbounds.day_hour >= start_date]

    if end_date:
        outbounds = outbounds[outbounds.day_hour <= end_date]
        inbounds = inbounds[inbounds.day_hour <= end_date]

    # Join outbounds and inbounds
    union_df = pd.merge(outbounds, inbounds, how='outer', on=[
                        'day_hour', 'station_id'], suffixes=('_outbound', '_inbound')).fillna(0)
    # Add weather data
    analysis_df = pd.merge(union_df, weather_df, how='left',
                           left_on='day_hour', right_index=True)

    analysis_df['weekday'] = analysis_df['day_hour'].dt.weekday
    analysis_df['hour'] = analysis_df['day_hour'].dt.hour
    analysis_df['day'] = analysis_df['day_hour'].dt.date
    analysis_df['year'] = analysis_df['day_hour'].dt.year
    analysis_df['month'] = analysis_df['day_hour'].dt.month
    analysis_df['day_of_month'] = analysis_df['day_hour'].dt.day

    columns_ordered = ['day_hour', 'station_id', 'weekday', 'day', 'hour', 'day_of_month', 'month', 'year', 'temperature',
                       'wind_speed', 'weather_phenomena', 'duration_outbound', 'trips_outbound', 'duration_inbound', 'trips_inbound']

    return analysis_df[columns_ordered]


weather_phen_dict = {
    'no phenomena': 0,
    'rain': 1,
    'fog': 2,
    'snow': 3
}


def enhance_station_for_prediction(id, stations_df, weather_df, start_date=None, end_date=None):

    station = stations_df[stations_df.station_id == id][['day_hour', 'duration', 'trips']].sort_values(
        by='day_hour').set_index('day_hour').resample('1h').agg({'duration': 'mean', 'trips': 'mean'}).fillna(0)

    station['weekday'] = station.index.weekday
    station['year'] = station.index.year
    station['month'] = station.index.month
    station['day_of_month'] = station.index.day
    station['hour'] = station.index.hour
    station = station.merge(weather_df, how='left',
                            left_index=True, right_index=True)

    station = station.replace({'weather_phenomena': weather_phen_dict})

    columns = ['trips', 'day_of_month', 'month', 'year', 'hour', 'duration',
               'weekday', 'temperature', 'wind_speed', 'weather_phenomena']
    station = station[columns]

    if start_date:
        station = station.loc[start_date:]

    if end_date:
        station = station.loc[:end_date]

    return station


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
