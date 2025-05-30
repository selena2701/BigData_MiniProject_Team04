import streamlit as st
import pandas as pd
import numpy as np
import streamlit_echarts as st_echarts

st.set_page_config(page_title="Customer Segmentation Insights", layout="wide")

st.title("Customer Segmentation Insights Dashboard")

# Load mined data
try:
    cs_df = pd.read_csv("Source_Code/mined_results.csv", parse_dates=['InvoiceDate'])
except FileNotFoundError:
    st.error("⚠️ Mined results file not found. Please ensure 'Source_Code/mined_results.csv' exists.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
country_list = cs_df['Country'].unique().tolist()
selected_country = st.sidebar.multiselect("Select Country", country_list, default=country_list)

min_date, max_date = cs_df['InvoiceDate'].min(), cs_df['InvoiceDate'].max()
selected_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Apply filters safely
if isinstance(selected_date, list) and len(selected_date) == 2:
    start_date, end_date = pd.to_datetime(selected_date[0]), pd.to_datetime(selected_date[1])
else:
    start_date, end_date = min_date, max_date

filtered_df = cs_df[(cs_df['Country'].isin(selected_country)) & (cs_df['InvoiceDate'] >= start_date) & (cs_df['InvoiceDate'] <= end_date)]

st.subheader("Filtered Data Overview")
if filtered_df.empty:
    st.warning("No data matches the selected filters.")
else:
    st.write(filtered_df)

    def dynamic_height(x_data):
        base_height = 500
        extra_height = (len(x_data) // 10) * 100
        return f"{base_height + extra_height}px"

    def build_bar_option(x_data, y_data, title):
        return {
            "title": {"text": title, "left": "center"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"bottom": 120},
            "xAxis": {"type": "category", "data": x_data, "axisLabel": {"interval": 0, "rotate": 90}},
            "yAxis": {"type": "value"},
            "series": [{"data": y_data, "type": "bar"}]
        }

    # Sales by Country
    country_sales = filtered_df.groupby("Country").amount.sum().sort_values(ascending=False)
    st.subheader("Amount Sales by Country")
    option = build_bar_option(country_sales.index.tolist(), country_sales.values.tolist(), "Amount Sales by Country")
    st_echarts.st_echarts(option, height=dynamic_height(country_sales))
    st.markdown("---")

    # Product-based charts
    AmoutSum = filtered_df.groupby(["Description"]).amount.sum().sort_values(ascending=False)
    inv = filtered_df.groupby(["Description"]).InvoiceNo.nunique().sort_values(ascending=False)

    Top10 = list(AmoutSum[:10].index)
    st.subheader("Top 10 Products in Sales Amount")
    option_top10 = build_bar_option(Top10, AmoutSum[Top10].values.tolist(), "Top 10 Products in Sales Amount")
    st_echarts.st_echarts(option_top10, height=dynamic_height(Top10))
    st.markdown("---")

    Top10Ev = list(inv[:10].index)
    st.subheader("Events of Top 10 Most Sold Products")
    option_top10ev = build_bar_option(Top10Ev, inv[Top10Ev].values.tolist(), "Top 10 Most Sold Products (Event Count)")
    st_echarts.st_echarts(option_top10ev, height=dynamic_height(Top10Ev))
    st.markdown("---")

    Top15ev = list(inv[:15].index)
    st.subheader("Sales Amount of Top 15 Most Sold Products")
    option_top15ev = build_bar_option(Top15ev, AmoutSum[Top15ev].sort_values(ascending=False).values.tolist(), "Top 15 Most Sold Products (Sales Amount)")
    st_echarts.st_echarts(option_top15ev, height=dynamic_height(Top15ev))
    st.markdown("---")

    Top50 = list(AmoutSum[:50].index)
    st.subheader("Top 50 Products in Sales Amount")
    option_top50 = build_bar_option(Top50, AmoutSum[Top50].values.tolist(), "Top 50 Products in Sales Amount")
    st_echarts.st_echarts(option_top50, height=dynamic_height(Top50))
    st.markdown("---")

    Top50Ev = list(inv[:50].index)
    st.subheader("Top 50 Most Sold Products (Event Count)")
    option_top50ev = build_bar_option(Top50Ev, inv[Top50Ev].values.tolist(), "Top 50 Most Sold Products (Event Count)")
    st_echarts.st_echarts(option_top50ev, height=dynamic_height(Top50Ev))
    st.markdown("---")
