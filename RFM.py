# Creating customer Segmentation by using RFM
#   Recency, Frequency, Monetary
# Check the image for better understanding of the Segments
# RF segmentation and RFM segmentation

# RFM - Costumer Segmentation with RFM

# Business Problem
# Data Understanding
# Data Preparation
# Calculating RFM Metrics
# Calculating RFM Scores
# Creating & Analysing RFM Segments

##################################################################
# 1 - Business Problem

# Online Sales Marketing Company wants to segment its customers
# In order to create new marketing campaigns

#################################################################

# Dataframe
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# A real online retail transaction data set of two years.

################################################################

# Data Understanding:

# InvoiceNo: Invoice number for each unique sales. If begins with C means canceled
# StockCode: Item codes (unique for each)
# Description: Information of the items (describes what is the item)
# Quantity: Quantity of the items which are sold, shown in the Invoice
# InvoiceDate: Invoice date - day month year and hour with sec
# UnitPrice: Price of the items which are sold, shown by GBP
# CustomerID: Customer number - unique for each
# Country: Shows the country items are sold

##################################################################
# Importing

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.use('Qt5Agg')

##################################################################
# Settings

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

##################################################################
# Data Read

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()

##################################################################
# Data Preparation

df.isnull().any()
# Description and Customer ID return True
#   means there are Null values in Desc and C ID

df.dropna(inplace=True)

##################################################################
# Data Preparation
##################################################################
# Cancelled Orders

# Invoice starts with C means Cancelled Order
#   Cancelled Order should be deleted
#   I will consider cancelled orders as if returns and will not delete

df[df["Invoice"].str.contains("C", na=False)].shape # 9839 rows shows cancelled orders
# df = df[~df["Invoice"].str.contains("C", na=False)]

##################################################################
# Total_Price

df["Total_Price"] = df["Quantity"] * df["Price"] # creating new columns as Total_price
df[df["Total_Price"] < 0].shape # 9839 rows shows Total_price below 0

# Can check Total_Price by invoice quickly
#   There are multiple rows for 1 invoice, because multiple items are sold in 1 invoice
#       By grouping invoice, we can check the total price for each invoice
df.groupby("Invoice").agg({"Total_Price": "sum"}).head()

##################################################################
# Date time for Analysis Date
#   Last invoice date may be 2 months ago or even more
#       That's the reason of creating an analysis date

df["InvoiceDate"].max() # 2010-12-09
today_date = dt.datetime(2010, 12, 11)

##################################################################
# RFM
##################################################################
# Get RFM Columns
# Recency how many date has been after last purchase per customer
# Frequency what is the total invoice number per customer
# Monetary total paid per customer
# Per customer is important that's why grouping by customer id

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                     "Invoice": lambda num: num.nunique(),
                                     "Total_Price": lambda price: price.sum()})

rfm.columns = ["recency", "frequency", "monetary"] # change the columns names as r f m
rfm[rfm["monetary"] < 0].shape # 92 rows below 0, not 9839 !!!!!

rfm = rfm[rfm["monetary"] > 0] # monetary can't be below 0
# monetary is not the total price per items, it's the total price per customer.
#   if monetary is below 0, company owes to customer !!!

##################################################################
# Max values correction for monetary
rfm.describe().T


def outliers(dataframe, variable):
    """

    Parameters
    q1: getting average of min value by checking first %1 quartile values
    q3: getting average of max value by checking last %99 quartile values

    Parameters are may vary acc to dataframe, on this dataframe min values are not
        far away from each other thats why %1 is choosed, and max values are also
        not far away from each other only couple of outliers are in dataframe, this
        is the reason why %99 is choosed. Check the dataframe !!!!
    ----------
    dataframe: dataframe which contains all the data as csv or excel or anyother
    variable: columns which contain the values for the quartiles

    Returns

    Returns are lower limit and upper limit which will be used in another function
        in order to make equation the min and max outliers to min average and max
        average values which is derived from up_limit and low_limit formulas
        check replace_with_thresholds function !!!
    -------

    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = (quartile3 - quartile1)
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """

    Parameters

    low_limit and up_limit are derived from outliers func
    ----------
    dataframe: dataframe which contain all the data as csv or excel or anyother
    variable: columns that will be replaced with up_limit and low_limit

    Returns
    -------

    """
    low_limit, up_limit = outliers(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(rfm, "monetary")

##################################################################
# RF and RFM Scores

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# why recency goes from 5 to 1 and not the others
#   do not forget recency shows how recently customer purchased
#       that's why if recency is smallest it's the highest score (5) for customer
# if frequency is low so score is also low
# if monetary is low so score is also low

rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                   rfm["frequency_score"].astype(str))

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str) +
                    rfm["monetary_score"].astype(str))

# astype(str) makes integers to string, so scores can be \
# written as 555

##################################################################
# Segmentation 1

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

##################################################################
# RFM dataframe reduce columns
rfm.head()
rfm = rfm[["recency", "frequency", "monetary", "RF_SCORE", "RFM_SCORE", "segment"]]

# grouping by segmentation to checking recency, frequency and monetary
rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "sum"})

##################################################################
# Segmentation 2

nhp = ["514", "515", "524", "525", "545"] # new and high paid customers

new_high_paid = rfm[rfm["RFM_SCORE"].isin(nhp)]

from numpy import mean
sns.barplot(x="segment", y="monetary", data=new_high_paid, estimator=mean)

#################################

hp = ["434", "435", "443", "444", "445", "453", "454", "455", \
      "533", "534", "535", "543", "544", "553", "554"]

high_paid = rfm[rfm["RFM_SCORE"].isin(hp)]

from numpy import mean
sns.barplot(x="segment", y="monetary", data=high_paid, estimator=mean)

#################################

php = ["332", "333", "334", "335", "342", "343", "34", "345",\
       "352", "353", "354", "355", "432", "443", "452", "532",\
       "542", "552"]

potential_high_paid = rfm[rfm["RFM_SCORE"].isin(php)]

from numpy import mean
sns.barplot(x="segment", y="monetary", data=potential_high_paid, estimator=mean)
##################################################################
# Getting EXCEL for top 3 segment customers

new_high_paid.to_excel("new_high_paid_customers.xlsx")
high_paid.to_excel("high_paid_customers.xlsx")
potential_high_paid.to_excel("potential_high_paid_customers.xlsx")

##################################################################
abl = ["211", "212","213", "214", "215", "221", "222", "223",
       "224", "225", "231", "232", "233", "234", "235", "241",
       "242", "243", "244", "245", "251", "252", "253", "254",
       "255", "311", "312", "315", "321", "322", "325", "331",
       "341", "351"]

about_to_leave = rfm[rfm["RFM_SCORE"].isin(abl)]

about_to_leave.to_excel("about_to_leave_customers.xlsx")















