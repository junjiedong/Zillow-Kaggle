import numpy as np
import pandas as pd
import gc

"""
    Load training data from csv file
"""
def load_training_data(file_name):
    return pd.read_csv(file_name)


"""
    Load house properties data
"""
def load_properties_data(file_name):

    # Helper function for parsing the flag attributes
    def convert_true_to_float(df, col):
        df.loc[df[col] == 'true', col] = '1'
        df.loc[df[col] == 'Y', col] = '1'
        df[col] = df[col].astype(float)

    prop = pd.read_csv(file_name, dtype={
        'propertycountylandusecode': str,
        'hashottuborspa': str,
        'propertyzoningdesc': str,
        'fireplaceflag': str,
        'taxdelinquencyflag': str
    })

    for col in ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']:
        convert_true_to_float(prop, col)

    return prop


"""
    Assign better names to all feature columns of 'properties' table
"""
def rename_columns(prop):
    prop.rename(columns={
        'parcelid': 'parcelid',  # Unique identifier of parcels
        'airconditioningtypeid': 'cooling_id',  # type of cooling system (if any), 1~13
        'architecturalstyletypeid': 'architecture_style_id',  # Architectural style of the home, 1~27
        'basementsqft': 'basement_sqft',  # Size of the basement
        'bathroomcnt': 'bathroom_cnt',  # Number of bathrooms (including fractional bathrooms)
        'bedroomcnt': 'bedroom_cnt',  # Number of bedrooms
        'buildingclasstypeid': 'framing_id',  # The building framing type, 1~5
        'buildingqualitytypeid': 'quality_id',  # building condition from best (lowest) to worst (highest)
        'calculatedbathnbr': 'bathroom_cnt_calc',  # Same meaning as 'bathroom_cnt'?
        'decktypeid': 'deck_id',  # Type of deck (if any)
        'finishedfloor1squarefeet': 'floor1_sqft',  # Size of finished living area on first floor
        'calculatedfinishedsquarefeet': 'finished_area_sqft_calc',  # calculated total finished living area
        'finishedsquarefeet12': 'finished_area_sqft',  # Same meaning as 'finished_area_sqft_calc'?
        'finishedsquarefeet13': 'perimeter_area',  # Perimeter living area
        'finishedsquarefeet15': 'total_area',  # Total area
        'finishedsquarefeet50': 'floor1_sqft_unk',  # Same meaning as 'floor1_sqft'?
        'finishedsquarefeet6': 'base_total_area',  # Base unfinished and finished area
        'fips': 'fips',  # Federal Information Processing Standard code
        'fireplacecnt': 'fireplace_cnt',  # Number of fireplaces in the home (if any)
        'fullbathcnt': 'bathroom_full_cnt',  # Number of full bathrooms
        'garagecarcnt': 'garage_cnt',  # Total number of garages
        'garagetotalsqft': 'garage_sqft',  # Total size of the garages
        'hashottuborspa': 'spa_flag',  # Whether the home has a hot tub or spa
        'heatingorsystemtypeid': 'heating_id',  # type of heating system, 1~25
        'latitude': 'latitude',  # latitude of the middle of the parcel multiplied by 1e6
        'longitude': 'longitude',  # longitude of the middle of the parcel multiplied by 1e6
        'lotsizesquarefeet': 'lot_sqft',  # Area of the lot in sqft
        'poolcnt': 'pool_cnt', # Number of pools in the lot (if any)
        'poolsizesum': 'pool_total_size',  # Total size of the pools
        'pooltypeid10': 'pool_unk_1',
        'pooltypeid2': 'pool_unk_2',
        'pooltypeid7': 'pool_unk_3',
        'propertycountylandusecode': 'county_landuse_code',
        'propertylandusetypeid': 'landuse_type_id' ,  # Type of land use the property is zoned for, 25 categories
        'propertyzoningdesc': 'zoning_description',  # Allowed land uses (zoning) for that property
        'rawcensustractandblock': 'census_1',
        'regionidcity': 'city_id',  # City in which the property is located (if any)
        'regionidcounty': 'county_id',  # County in which the property is located
        'regionidneighborhood': 'neighborhood_id',  # Neighborhood in which the property is located
        'regionidzip': 'region_zip',
        'roomcnt': 'room_cnt',  # Total number of rooms in the principal residence
        'storytypeid': 'story_id',  # Type of floors in a multi-story house, 1~35
        'threequarterbathnbr': 'bathroom_small_cnt',  # Number of 3/4 bathrooms
        'typeconstructiontypeid': 'construction_id',  # Type of construction material, 1~18
        'unitcnt': 'unit_cnt',  # Number of units the structure is built into (2=duplex, 3=triplex, etc)
        'yardbuildingsqft17': 'patio_sqft',  # Patio in yard
        'yardbuildingsqft26': 'storage_sqft',  # Storage shed/building in yard
        'yearbuilt': 'year_built',  # The year the principal residence was built
        'numberofstories': 'story_cnt',  # Number of stories or levels the home has
        'fireplaceflag': 'fireplace_flag',  # Whether the home has a fireplace
        'structuretaxvaluedollarcnt': 'tax_structure',
        'taxvaluedollarcnt': 'tax_parcel',
        'assessmentyear': 'tax_year',  # The year of the property tax assessment (2015 for 2016 data)
        'landtaxvaluedollarcnt': 'tax_land',
        'taxamount': 'tax_property',
        'taxdelinquencyflag': 'tax_overdue_flag',  # Property taxes are past due as of 2015
        'taxdelinquencyyear': 'tax_overdue_year',  # Year for which the unpaid propert taxes were due
        'censustractandblock': 'census_2'
    }, inplace=True)


"""
    Convert some categorical variables to 'category' type
    Convert float64 variables to float32
    Note: In LightGBM, negative integer value for a categorical feature will be treated as missing value
"""
def retype_columns(prop):

    def float_to_categorical(df, col):
        df[col] = df[col] - df[col].min()  # Convert the categories to have smaller labels (start from 0)
        df.loc[df[col].isnull(), col] = -1
        df[col] = df[col].astype(int).astype('category')

    list_float2categorical = ['cooling_id', 'architecture_style_id', 'framing_id', 'quality_id',
                             'heating_id', 'county_id', 'construction_id', 'fips', 'landuse_type_id']

    # Convert categorical variables to 'category' type, and float64 variables to float32
    for col in prop.columns:
        if col in list_float2categorical:
            float_to_categorical(prop, col)
        elif prop[col].dtype.name == 'float64':
            prop[col] = prop[col].astype(np.float32)

    gc.collect()


"""
    Drop features that are not useful or too messy
"""
def drop_features(features):
    unused_feature_list = ['parcelid', 'logerror']  # not features
    unused_feature_list += ['county_landuse_code', 'zoning_description']  # cannot be used directly

    # too many missing (LightGBM is robust against bad/unrelated features, so probably no need to drop)
    missing_list = ['framing_id', 'construction_id', 'deck_id', 'pool_unk_1', 'architecture_style_id',
                'story_id', 'perimeter_area', 'pool_total_size', 'basement_sqft', 'storage_sqft', 'fireplace_flag']
    # unused_feature_list += missing_list

    return features.drop(unused_feature_list, axis=1, errors='ignore')


"""
    Compute and return datetime aggregate feature tables from a training set
    The returned tables can be joined for both training and inference
"""
def compute_datetime_aggregate_features(train):
    # Add temporary year/month/quarter columns
    dt = pd.to_datetime(train.transactiondate).dt
    train['year'] = dt.year
    train['month'] = dt.month
    train['quarter'] = dt.quarter

    # Median logerror within the category
    logerror_year = train.groupby('year').logerror.median().to_frame() \
                                .rename(index=str, columns={"logerror": "logerror_year"})
    logerror_month = train.groupby('month').logerror.median().to_frame() \
                                .rename(index=str, columns={"logerror": "logerror_month"})
    logerror_quarter = train.groupby('quarter').logerror.median().to_frame() \
                                .rename(index=str, columns={"logerror": "logerror_quarter"})

    logerror_year.index = logerror_year.index.map(int)
    logerror_month.index = logerror_month.index.map(int)
    logerror_quarter.index = logerror_quarter.index.map(int)

    # Drop the temporary columns
    train.drop(['year', 'month', 'quarter'], axis=1, errors='ignore', inplace=True)

    return logerror_year, logerror_month, logerror_quarter


"""
    Add aggregrate datetime features to a feature table
    The input table needs to have a 'transactiondate' columns
    The 'transactiondate' column is deleted from the table in the end
"""
def add_datetime_aggregate_features(df, logerror_year, logerror_month, logerror_quarter):
    # Add temporary year/month/quarter columns
    dt = pd.to_datetime(df.transactiondate).dt
    df['year'] = dt.year
    df['month'] = dt.month
    df['quarter'] = dt.quarter

    # Join the aggregate features
    df = df.merge(how='left', right=logerror_year, on='year')
    df = df.merge(how='left', right=logerror_month, on='month')
    df = df.merge(how='left', right=logerror_quarter, on='quarter')

    # Drop the temporary columns
    df = df.drop(['year', 'month', 'quarter', 'transactiondate'], axis=1, errors='ignore')
    return df


"""
    Look at how complete (i.e. no missing value) each feature is
"""
def print_complete_percentage(df):
    complete_percent = []
    total_cnt = len(df)
    for col in df.columns:
        complete_cnt = total_cnt - (df[col].isnull()).sum()
        complete_cnt -= (df[col] == -1).sum()
        complete_percent.append((col, complete_cnt * 1.00 / total_cnt))
    complete_percent.sort(key=lambda x: x[1], reverse=True)
    for col, percent in complete_percent:
        print("{}: {}".format(col, percent))
