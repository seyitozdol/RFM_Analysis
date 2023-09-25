# **************************
# Business Problem:
# FLO, an online shoe store, wants to segment its customers and determine marketing strategies based on these segments.
# To this end, customer behaviors will be identified and groups will be formed based on expected patterns in these behaviors.
# **************************

# **************************
# Dataset Story:
# The dataset consists of information obtained from the past shopping behaviors of customers who shopped at FLO
# through OmniChannel (both online and offline) in the years 2020-2021."
# **************************

# **************************
# VARIABLES
# master_id: Unique customer number
# order_channel: The channel used for shopping, indicating the platform (Android, ios, Desktop, Mobile)
# last_order_channel: The channel where the most recent shopping was done
# first_order_date: The date of the customer's first purchase
# last_order_date: The date of the customer's last purchase
# last_order_date_online: The date of the customer's last purchase on an online platform
# last_order_date_offline: The date of the customer's last purchase on an offline platform
# order_num_total_ever_online: Total number of purchases the customer has made on an online platform
# order_num_total_ever_offline: Total number of purchases the customer has made offline
# customer_value_total_ever_offline: Total amount spent by the customer on offline purchases
# customer_value_total_ever_online: Total amount spent by the customer on online purchases
# interested_in_categories_12: List of categories the customer has shopped in over the last 12 months
# **************************

# **************************
# Project Tasks
# Task 1: Calculate the RFM Metrics
# Task 2: Calculate the RF Score
# Task 3: Define the RF Score as a Segment
# Task 4: Save the specific customers given the details below
# Task 5: Save the specific customers given the details below


import datetime as dt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 500)
# pd.set_option('display_max_rows',None)
pd.set_option('display.float_format',lambda x : '%.5f' % x)

df_ = pd.read_csv('flo_data_20K.csv')
df = df_.copy()


def analyze_missing_values(df):
    na_cols = df.columns[df.isna().any()].tolist()
    total_missing = df[na_cols].isna().sum().sort_values(ascending=False)
    percentage_missing = ((df[na_cols].isna().sum() / df.shape[0]) * 100).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Count': total_missing, 'Percentage (%)': np.round(percentage_missing, 2)})
    return missing_data


# to get an initial understanding of the data's structure, its content, and if there are any missing values that need to be addressed.
def sum_df(dataframe, head=6):
    print("~~~~~~~~~~|-HEAD-|~~~~~~~~~~ ")
    print(dataframe.head(head))
    print("~~~~~~~~~~|-TAIL-|~~~~~~~~~~ ")
    print(dataframe.tail(head))
    print("~~~~~~~~~~|-TYPES-|~~~~~~~~~~ ")
    print(dataframe.dtypes)
    print("~~~~~~~~~~|-SHAPE-|~~~~~~~~~~ ")
    print(dataframe.shape)
    print("~~~~~~~~~~|-NUMBER OF UNIQUE-|~~~~~~~~~~ ")
    print(dataframe.nunique())
    print("~~~~~~~~~~|-NA-|~~~~~~~~~~ ")
    print(dataframe.isnull().sum())
    print("~~~~~~~~~~|-QUANTILES-|~~~~~~~~~~ ")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("~~~~~~~~~~|-NUMERIC COLUMNS-|~~~~~~~~~~ ")
    print([i for i in dataframe.columns if dataframe[i].dtype != "O"])
    print("~~~~~~~~~~|-MISSING VALUE ANALYSIS-|~~~~~~~~~~ ")
    print(analyze_missing_values(dataframe))

sum_df(df)

#**********************************************************************************
# Task 1: Calculate the RFM Metrics
#**********************************************************************************

def calculate_rfm_values(dataframe, master_id_col, last_order_date_col, total_order_col, total_price_col):
    today_date = dataframe[last_order_date_col].max() + dt.timedelta(days=2)

    rfm = dataframe.groupby(master_id_col).agg({
        last_order_date_col: lambda date: (today_date - date.max()).days,
        total_order_col: lambda x: x.sum(),
        total_price_col: lambda x: x.sum()
    }).reset_index()

    rfm.columns = ['master_id', 'recency', 'frequency', 'monetary']
    return rfm
#*************************************
# Task 2: Calculate the RF Score
#*************************************



def assign_rfm_scores(rfm):
    rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    return rfm



#***********************************************
# Task 3: Define the RF Score as a Segment
#***********************************************

def map_rfm_to_segment(rfm):
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['RF_SCORE'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
    return rfm


def custom_rfm(dataframe, master_id_col='master_id', last_order_date_col='last_order_date',
               total_order_col='Total_Order', total_price_col='Total_Price'):
    dataframe = dataframe.copy()
    dataframe['Total_Order'] = dataframe['order_num_total_ever_online'] + dataframe['order_num_total_ever_offline']
    dataframe['Total_Price'] = dataframe['customer_value_total_ever_offline'] + dataframe[
        'customer_value_total_ever_online']
    dataframe = dataframe.astype({col: 'datetime64[ns]' for col in dataframe.columns if 'date' in col})

    rfm = calculate_rfm_values(dataframe, master_id_col, last_order_date_col, total_order_col, total_price_col)
    rfm = assign_rfm_scores(rfm)
    rfm = map_rfm_to_segment(rfm)

    return rfm

rfm = custom_rfm(df)

#*********************************************************************************
# Task 4: FLO is incorporating a new women's shoe brand into its inventory.
# The product prices of the introduced brand are generally above the preferences of most customers.
# For this reason, it is desired to specifically reach out to customers who would be interested in this brand's promotion and product sales.
# Dedicated customers (champions, loyal_customers) and those who shop in the women's category will specifically be contacted.
# Save the IDs of these customers to a CSV file.
#*********************************************************************************


dedicated_customers = (rfm[(rfm["segment"]=="champions") | (rfm["segment"]=="loyal_customers")])

women_category = df[(df["interested_in_categories_12"]).str.contains("KADIN")]

dedicated_women_category = pd.merge(dedicated_customers,women_category[["interested_in_categories_12","master_id"]],on=["master_id"])

dedicated_women_category = dedicated_women_category.drop(dedicated_women_category.loc[:, 'recency':'interested_in_categories_12'].columns, axis=1)

dedicated_women_category.to_csv("dedicated_women_category_info.csv")


#*********************************************************************************
# Task 5: There are plans to offer nearly 40% discounts on Men's and Children's products.
# Customers who have been good in the past but haven't shopped for a long time (those who need to be retained),
# dormant ones, and new customers, who might be interested in these discount categories, are specifically targeted.
# Save the IDs of suitable profile customers to a CSV file.
#*********************************************************************************

specific_profile = rfm[(rfm["segment"]=="cant_loose") | (rfm["segment"]=="about_to_sleep") | (rfm["segment"]=="new_customers")]

boys_and_men =df[(df["interested_in_categories_12"]).str.contains("ERKEK|COCUK")]

boys_and_men_profile = pd.merge(specific_profile,boys_and_men[["interested_in_categories_12","master_id"]],on=["master_id"])

boys_and_men_profile = boys_and_men_profile.drop(boys_and_men_profile.loc[:,'recency':'interested_in_categories_12'].columns, axis=1)

boys_and_men_profile.to_csv("boys_and_men_customer_profile.csv")

