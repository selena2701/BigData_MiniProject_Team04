import streamlit as st
import pandas as pd
import numpy as np

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

    def show_vega_bar_chart(df, x_col, y_col, title, height=500):
        st.vega_lite_chart(
            df,
            {
                "mark": {"type": "bar", "tooltip": True},
                "encoding": {
                    "x": {"field": x_col, "type": "nominal", "axis": {"labelAngle": -90}},
                    "y": {"field": y_col, "type": "quantitative"},
                    "tooltip": [{"field": x_col, "type": "nominal"}, {"field": y_col, "type": "quantitative"}]
                },
                "title": title,
                "height": height
            },
            use_container_width=True
        )

    # Sales by Country
    country_sales = filtered_df.groupby("Country").amount.sum().sort_values(ascending=False).reset_index()
    st.subheader("Amount Sales by Country")
    show_vega_bar_chart(country_sales, "Country", "amount", "Amount Sales by Country")
    st.markdown("---")

    # Product-based charts
    AmoutSum = filtered_df.groupby(["Description"]).amount.sum().sort_values(ascending=False).reset_index()
    inv = filtered_df.groupby(["Description"]).InvoiceNo.nunique().sort_values(ascending=False).reset_index()

    Top10 = AmoutSum.head(10)
    st.subheader("Top 10 Products in Sales Amount")
    show_vega_bar_chart(Top10, "Description", "amount", "Top 10 Products in Sales Amount")
    st.markdown("---")

    Top10Ev = inv.head(10)
    st.subheader("Events of Top 10 Most Sold Products")
    show_vega_bar_chart(Top10Ev, "Description", "InvoiceNo", "Top 10 Most Sold Products (Event Count)")
    st.markdown("---")

    Top15ev = AmoutSum.merge(inv, on="Description").head(15)
    st.subheader("Sales Amount of Top 15 Most Sold Products")
    show_vega_bar_chart(Top15ev, "Description", "amount", "Top 15 Most Sold Products (Sales Amount)")
    st.markdown("---")

    Top50 = AmoutSum.head(50)
    st.subheader("Top 50 Products in Sales Amount")
    show_vega_bar_chart(Top50, "Description", "amount", "Top 50 Products in Sales Amount")
    st.markdown("---")

    Top50Ev = inv.head(50)
    st.subheader("Top 50 Most Sold Products (Event Count)")
    show_vega_bar_chart(Top50Ev, "Description", "InvoiceNo", "Top 50 Most Sold Products (Event Count)")
    st.markdown("---")
