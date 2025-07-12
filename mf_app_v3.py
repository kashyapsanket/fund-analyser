import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
import plotly.express as px
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools

# --- Page Configuration ---
st.set_page_config(
    page_title="Portfolio Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Show verbose error details in the app
st.set_option('client.showErrorDetails', True)

# --- Database Connection ---
@st.cache_resource
def get_engine():
    """Creates a SQLAlchemy engine to connect to the SQLite database."""
    db_path = 'portfolio.db'
    if not os.path.exists(db_path):
        st.error(f"Database file '{db_path}' not found. Please run the database creation script first.")
        st.stop()
    return create_engine(f'sqlite:///{db_path}')

# --- Data Loading Functions ---
@st.cache_data
def load_all_data(_engine):
    """
    Loads all necessary data from the database and standardizes column names.
    """
    with _engine.connect() as connection:
        # Assuming the updated funds table is named 'funds' and has 'fund_code'
        funds_df = pd.read_sql('SELECT fund_id, amc_name, fund_name, fund_code FROM funds_v2', connection)
        equities_df = pd.read_sql('SELECT isin, company_name, industry_rating, market_cap_type FROM equities_v3', connection)
        holdings_df = pd.read_sql('SELECT fund_id, company_name, category, pct_net_assets, industry_rating, isin FROM portfolio_holdings_v2', connection)

    # Convert fund_code to integer type, handling potential nulls
    funds_df['fund_code'] = pd.to_numeric(funds_df['fund_code'], errors='coerce').astype('Int64')
    # Standardize ISIN column to lowercase for consistent merging
    if 'ISIN' in equities_df.columns:
        equities_df.rename(columns={'ISIN': 'isin'}, inplace=True)


    return funds_df, equities_df, holdings_df

# --- API and Performance Calculation Functions ---
@st.cache_data(show_spinner=False) # Hide spinner from this cached function
def get_fund_performance(fund_code):
    """Fetches and calculates historical performance for a given fund code."""
    if pd.isna(fund_code):
        return None, None
    try:
        url = f"https://api.mfapi.in/mf/{int(fund_code)}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == "SUCCESS":
            nav_data = pd.DataFrame(data['data'])
            nav_data['nav'] = pd.to_numeric(nav_data['nav'], errors='coerce')
            nav_data['date'] = pd.to_datetime(nav_data['date'], format='%d-%m-%Y')
            nav_data = nav_data.sort_values('date').reset_index(drop=True)
            return calculate_returns(nav_data)
        else:
            return None, None
    except (requests.exceptions.RequestException, ValueError):
        return None, None

def find_closest_past_date(df, target_date):
    """Finds the closest date in the dataframe on or before the target date."""
    past_dates = df[df['date'] <= target_date]
    if not past_dates.empty:
        return past_dates.iloc[-1]
    return None

def calculate_returns(nav_df):
    """Calculates returns for various periods from a NAV dataframe."""
    if nav_df.empty or len(nav_df) < 2:
        return {}, None

    latest_record = nav_df.iloc[-1]
    latest_nav = latest_record['nav']
    latest_date = latest_record['date']
    
    returns = {}
    periods = {
        '1M': 1, '3M': 3, '6M': 6, '1Y': 12, '3Y': 36, 
        '5Y': 60, '10Y': 120
    }

    for label, months in periods.items():
        target_date = latest_date - relativedelta(months=months)
        start_record = find_closest_past_date(nav_df, target_date)
        
        if start_record is not None:
            start_nav = start_record['nav']
            num_years = months / 12.0
            
            if num_years <= 1: # Absolute return
                returns[label] = ((latest_nav / start_nav) - 1) * 100
            else: # Annualized return (CAGR)
                returns[label] = (((latest_nav / start_nav) ** (1 / num_years)) - 1) * 100
        else:
            returns[label] = None

    # Since Inception
    first_record = nav_df.iloc[0]
    first_nav = first_record['nav']
    first_date = first_record['date']
    total_years = (latest_date - first_date).days / 365.25
    if total_years > 0:
        returns['Since Inception'] = (((latest_nav / first_nav) ** (1 / total_years)) - 1) * 100
    else:
        returns['Since Inception'] = None

    return returns, latest_date

def calculate_overlap_matrix(portfolio_funds, all_holdings):
    """Calculates the overlap matrix for a list of funds."""
    fund_names = [f['fund_name'] for f in portfolio_funds]
    fund_ids = [f['fund_id'] for f in portfolio_funds]
    
    overlap_matrix = pd.DataFrame(index=fund_names, columns=fund_names, dtype=float)
    
    equity_holdings = all_holdings[all_holdings['category'] == 'Equity'].copy()
    equity_holdings['pct_net_assets'] = pd.to_numeric(equity_holdings['pct_net_assets'], errors='coerce').fillna(0)

    for i in range(len(fund_ids)):
        for j in range(i, len(fund_ids)):
            fund_a_name = fund_names[i]
            fund_b_name = fund_names[j]
            
            if i == j:
                overlap_matrix.loc[fund_a_name, fund_b_name] = 100.0
                continue

            fund_a_id = fund_ids[i]
            fund_b_id = fund_ids[j]

            fund_a_holdings = equity_holdings[equity_holdings['fund_id'] == fund_a_id][['isin', 'pct_net_assets']]
            fund_b_holdings = equity_holdings[equity_holdings['fund_id'] == fund_b_id][['isin', 'pct_net_assets']]

            common_holdings = pd.merge(fund_a_holdings, fund_b_holdings, on='isin', suffixes=('_a', '_b'))

            if not common_holdings.empty:
                common_holdings['min_pct'] = common_holdings[['pct_net_assets_a', 'pct_net_assets_b']].min(axis=1)
                overlap_pct = common_holdings['min_pct'].sum()
            else:
                overlap_pct = 0.0
            
            overlap_matrix.loc[fund_a_name, fund_b_name] = overlap_pct
            overlap_matrix.loc[fund_b_name, fund_a_name] = overlap_pct
            
    return overlap_matrix


def group_small_slices(df, value_col, name_col, threshold=2.0):
    """Groups small slices of a DataFrame into an 'Other' category for pie charts."""
    if df[value_col].sum() == 0:
        return df[[name_col, value_col]]
        
    df['percentage'] = (df[value_col] / df[value_col].sum()) * 100
    
    small_slices = df[df['percentage'] < threshold]
    
    if not small_slices.empty and len(small_slices) > 1:
        main_slices = df[df['percentage'] >= threshold]
        other_sum = small_slices[value_col].sum()
        other_row = pd.DataFrame({name_col: ['Other'], value_col: [other_sum]})
        
        grouped_df = pd.concat([main_slices[[name_col, value_col]], other_row], ignore_index=True)
        return grouped_df
    else:
        return df[[name_col, value_col]]

# --- Page Navigation and Rendering Functions ---

def go_to_page(page):
    st.session_state.page_name = page

def reset_app():
    """Clears all portfolio data and returns to the first page."""
    st.session_state.portfolio = []
    st.session_state.stock_portfolio = []
    go_to_page("Mutual Funds")

def render_mutual_funds_page(amc_list, selectable_funds_data):
    st.title("Step 1: Build Your Mutual Fund Portfolio")
    st.write("Select a fund, enter the amount you've invested, and add it to your portfolio.")
    st.divider()
    
    col1, col2, col3, col4 = st.columns([2, 3, 1.5, 1.5])
    with col1:
        selected_amc = st.selectbox("AMC Name", options=amc_list, index=None, placeholder="Select an AMC", key="mf_amc_select")
    with col2:
        if selected_amc:
            available_funds = selectable_funds_data[selectable_funds_data['amc_name'] == selected_amc]['fund_name'].tolist()
            selected_fund = st.selectbox("Fund Name", options=available_funds, index=None, placeholder="Select a Fund", key="mf_fund_select")
        else:
            st.selectbox("Fund Name", [], disabled=True, placeholder="First select an AMC", key="mf_fund_select_disabled")
            selected_fund = None
    with col3:
        invested_value = st.number_input("Invested Value (₹)", min_value=0.0, format="%.1f", step=1000.0, placeholder="Enter a value", key="mf_value")
    with col4:
        st.write("")
        st.write("")
        add_button = st.button("Add to Portfolio", type="primary", use_container_width=True, key="mf_add_button")
    
    if add_button:
        if selected_amc and selected_fund and invested_value > 0:
            fund_details = selectable_funds_data.loc[selectable_funds_data['fund_name'] == selected_fund].iloc[0]
            fund_id = fund_details['fund_id']
            fund_code = fund_details['fund_code']
            
            with st.spinner(f"Fetching performance for {selected_fund}..."):
                performance, latest_date = get_fund_performance(fund_code)

            new_entry = {
                "amc_name": selected_amc, 
                "fund_name": selected_fund, 
                "value": invested_value, 
                "fund_id": fund_id,
                "performance": performance,
                "latest_date": latest_date
            }
            st.session_state.portfolio.append(new_entry)
            st.success(f"Added {selected_fund} to your portfolio!")
        else:
            st.warning("Please fill in all fields before adding.")
    
    st.divider()
    st.subheader("Your Current Mutual Fund Portfolio")
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty.")
    else:
        for i in range(len(st.session_state.portfolio) - 1, -1, -1):
            item = st.session_state.portfolio[i]
            row_cols = st.columns([2, 3, 1.5, 0.5])
            row_cols[0].write(item['amc_name'])
            row_cols[1].write(item['fund_name'])
            # Format to one decimal place
            row_cols[2].write(f"₹{item['value']:,.1f}")
            if row_cols[3].button("🗑️", key=f"delete_mf_{i}"):
                st.session_state.portfolio.pop(i)
                st.rerun()
        
        total_value = sum(item['value'] for item in st.session_state.portfolio)
        st.markdown(f"### Total Mutual Fund Value: **₹{total_value:,.1f}**")

    st.divider()
    st.button("Proceed to Stocks →", on_click=go_to_page, args=("Stocks",), type="primary")

def render_stocks_page(equity_list):
    st.title("Step 2: Build Your Stock Portfolio")
    st.write("Select a stock, enter the value of your holding, and add it to your portfolio.")
    st.divider()
    
    col1, col2, col3 = st.columns([3, 1.5, 1.5])
    with col1:
        selected_stock = st.selectbox("Company Name", options=equity_list, index=None, placeholder="Select a stock", key="stock_select")
    with col2:
        stock_value = st.number_input("Invested Value (₹)", min_value=0.0, format="%.1f", step=1000.0, placeholder="Enter a value", key="stock_value")
    with col3:
        st.write("")
        st.write("")
        add_stock_button = st.button("Add to Portfolio", type="primary", use_container_width=True, key="stock_add_button")
    
    if add_stock_button:
        if selected_stock and stock_value > 0:
            st.session_state.stock_portfolio.append({"company_name": selected_stock, "value": stock_value})
            st.success(f"Added {selected_stock} to your portfolio!")
        else:
            st.warning("Please select a stock and enter a value.")
    
    st.divider()
    st.subheader("Your Current Stock Portfolio")
    if not st.session_state.stock_portfolio:
        st.info("Your stock portfolio is empty.")
    else:
        header_cols = st.columns([3, 1.5, 0.5])
        header_cols[0].markdown("**Company Name**")
        header_cols[1].markdown("**Invested Value (₹)**")
        for i in range(len(st.session_state.stock_portfolio) - 1, -1, -1):
            item = st.session_state.stock_portfolio[i]
            row_cols = st.columns([3, 1.5, 0.5])
            row_cols[0].write(item['company_name'])
            # Format to one decimal place
            row_cols[1].write(f"₹{item['value']:,.1f}")
            if row_cols[2].button("🗑️", key=f"delete_stock_{i}"):
                st.session_state.stock_portfolio.pop(i)
                st.rerun()
        total_stock_value = sum(item['value'] for item in st.session_state.stock_portfolio)
        # Format to one decimal place
        st.markdown(f"### Total Stock Value: **₹{total_stock_value:,.1f}**")
    
    st.divider()
    bcol1, bcol2 = st.columns([1,1])
    bcol1.button("← Back to Mutual Funds", on_click=go_to_page, args=("Mutual Funds",))
    bcol2.button("Proceed to Analysis →", on_click=go_to_page, args=("Analysis",), type="primary")

def render_track_record_table():
    """Renders the fund performance table with the latest date."""
    st.subheader("Fund Track Record (% Returns)")
    perf_data_items = [item for item in st.session_state.portfolio if item.get('performance')]
    
    if perf_data_items:
        latest_dates = [item['latest_date'] for item in perf_data_items if 'latest_date' in item and pd.notna(item.get('latest_date'))]
        if latest_dates:
            most_recent_date = max(latest_dates)
            st.caption(f"Data From - {most_recent_date.strftime('%d %B %Y')}")

        display_data = []
        for item in perf_data_items:
            row = {"Fund Name": item['fund_name']}
            row.update(item['performance'])
            display_data.append(row)
        perf_df = pd.DataFrame(display_data).set_index("Fund Name")
        st.dataframe(perf_df.style.format("{:.1f}%", na_rep="N/A"), use_container_width=True)

def render_analysis_page(equities_data, holdings_data):
    st.title("Step 3: Portfolio Analysis")
    
    bcol1, bcol2, bcol_spacer = st.columns([1.5, 1.5, 5])
    bcol1.button("← Back to Stocks", on_click=go_to_page, args=("Stocks",))
    bcol2.button("Reset Portfolio", on_click=reset_app)
    
    st.divider()

    if not st.session_state.portfolio and not st.session_state.stock_portfolio:
        st.warning("Your portfolio is empty. Please go back and add some funds or stocks.")
        st.stop()

    # --- Data Processing for Analysis ---
    direct_stocks_df = pd.DataFrame(st.session_state.stock_portfolio)
    if not direct_stocks_df.empty:
        direct_stocks_df = pd.merge(direct_stocks_df, equities_data, on="company_name", how="left")

    mf_portfolio_df = pd.DataFrame(st.session_state.portfolio)
    underlying_holdings_df = pd.DataFrame()
    if not mf_portfolio_df.empty:
        user_holdings = holdings_data[holdings_data['fund_id'].isin(mf_portfolio_df['fund_id'])]
        merged_holdings = pd.merge(user_holdings, mf_portfolio_df[['fund_id', 'value']], on='fund_id')
        merged_holdings['pct_net_assets'] = pd.to_numeric(merged_holdings['pct_net_assets'], errors='coerce').fillna(0)
        merged_holdings['absolute_value'] = (merged_holdings['pct_net_assets'] / 100) * merged_holdings['value']
        
        authoritative_equity_data = equities_data[['isin', 'industry_rating', 'market_cap_type']].drop_duplicates('isin')
        if 'industry_rating' in merged_holdings.columns:
            merged_holdings = merged_holdings.drop(columns=['industry_rating'])
        underlying_holdings_df = pd.merge(merged_holdings, authoritative_equity_data, on='isin', how='left')

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Stocks Only", "Mutual Funds Only", "Combined Portfolio", "Mutual Fund Overlap"])

    with tab1:
        st.header("Direct Stock Holdings Analysis")
        if direct_stocks_df.empty:
            st.info("You have not added any direct stocks to your portfolio.")
        else:
            col1, col2 = st.columns(2)
            sector_dist = direct_stocks_df.dropna(subset=['industry_rating']).groupby('industry_rating')['value'].sum().reset_index()
            grouped_sector_dist = group_small_slices(sector_dist, 'value', 'industry_rating')
            fig1 = px.pie(grouped_sector_dist, names='industry_rating', values='value', title='Sector Distribution')
            col1.plotly_chart(fig1, use_container_width=True)
            
            market_cap_dist = direct_stocks_df.dropna(subset=['market_cap_type']).groupby('market_cap_type')['value'].sum().reset_index()
            grouped_mc_dist = group_small_slices(market_cap_dist, 'value', 'market_cap_type')
            fig_mc1 = px.pie(grouped_mc_dist, names='market_cap_type', values='value', title='Market Cap Distribution')
            col2.plotly_chart(fig_mc1, use_container_width=True)
            
            st.divider()
            st.subheader("Top 10 Direct Stock Holdings")
            top_10_stocks = direct_stocks_df.sort_values('value', ascending=False).head(10)
            fig_top_stocks = px.bar(top_10_stocks, x='company_name', y='value', title='Top 10 Direct Stock Holdings', labels={'company_name': 'Company', 'value': 'Invested Value'})
            st.plotly_chart(fig_top_stocks, use_container_width=True)

    with tab2:
        st.header("Mutual Fund Holdings Analysis")
        if underlying_holdings_df.empty:
            st.info("You have not added any mutual funds to your portfolio.")
        else:
            col1, col2 = st.columns(2)
            amc_dist = mf_portfolio_df.groupby('amc_name')['value'].sum().reset_index()
            grouped_amc_dist = group_small_slices(amc_dist, 'value', 'amc_name')
            fig3 = px.pie(grouped_amc_dist, names='amc_name', values='value', title='Portfolio Distribution by AMC')
            col1.plotly_chart(fig3, use_container_width=True)
            
            asset_cat_dist = underlying_holdings_df.groupby('category')['absolute_value'].sum().reset_index()
            grouped_asset_dist = group_small_slices(asset_cat_dist, 'absolute_value', 'category')
            fig4 = px.pie(grouped_asset_dist, names='category', values='absolute_value', title='Underlying Asset Distribution')
            col2.plotly_chart(fig4, use_container_width=True)
            
            st.divider()
            st.subheader("Underlying Equity Analysis")
            underlying_equities = underlying_holdings_df[underlying_holdings_df['category'] == 'Equity']
            underlying_equities_no_total = underlying_equities[underlying_equities['company_name'].str.lower() != 'total']
            if not underlying_equities_no_total.empty:
                col3, col4 = st.columns(2)
                mf_sector_dist = underlying_equities_no_total.dropna(subset=['industry_rating']).groupby('industry_rating')['absolute_value'].sum().reset_index()
                grouped_mf_sector = group_small_slices(mf_sector_dist, 'absolute_value', 'industry_rating')
                fig5 = px.pie(grouped_mf_sector, names='industry_rating', values='absolute_value', title='Underlying Sector Distribution')
                col3.plotly_chart(fig5, use_container_width=True)
                
                mf_mc_dist = underlying_equities_no_total.dropna(subset=['market_cap_type']).groupby('market_cap_type')['absolute_value'].sum().reset_index()
                grouped_mf_mc = group_small_slices(mf_mc_dist, 'absolute_value', 'market_cap_type')
                fig_mc2 = px.pie(grouped_mf_mc, names='market_cap_type', values='absolute_value', title='Underlying Market Cap Distribution')
                col4.plotly_chart(fig_mc2, use_container_width=True)
                
                st.divider()
                st.subheader("Top 10 Underlying Stock Holdings")
                top_10_mf_stocks = underlying_equities_no_total.groupby('company_name')['absolute_value'].sum().sort_values(ascending=False).head(10).reset_index()
                fig_top_mf_stocks = px.bar(top_10_mf_stocks, x='company_name', y='absolute_value', title='Top 10 Underlying Stock Holdings in MFs', labels={'company_name': 'Company', 'absolute_value': 'Calculated Value'})
                st.plotly_chart(fig_top_mf_stocks, use_container_width=True)
            
            st.divider()
            render_track_record_table()

    with tab3:
        st.header("Combined Portfolio Analysis")
        direct_stocks_for_combo = pd.DataFrame()
        if not direct_stocks_df.empty:
            direct_stocks_for_combo = direct_stocks_df[['company_name', 'industry_rating', 'market_cap_type', 'value']].rename(columns={'value': 'absolute_value'})
        
        mf_equities_for_combo = pd.DataFrame()
        if not underlying_holdings_df.empty:
            mf_equities_for_combo = underlying_holdings_df[underlying_holdings_df['category'] == 'Equity']
            mf_equities_for_combo = mf_equities_for_combo[['company_name', 'industry_rating', 'market_cap_type', 'absolute_value']]
        if direct_stocks_for_combo.empty and mf_equities_for_combo.empty:
            st.info("No equity holdings found in your portfolio to combine.")
        else:
            combined_df = pd.concat([direct_stocks_for_combo, mf_equities_for_combo])
            combined_df_no_total = combined_df[combined_df['company_name'].str.lower() != 'total']
            total_equity_exposure = combined_df_no_total.groupby(['company_name', 'industry_rating', 'market_cap_type'])['absolute_value'].sum().reset_index()
            col1, col2 = st.columns(2)
            
            combined_sector_dist = total_equity_exposure.dropna(subset=['industry_rating']).groupby('industry_rating')['absolute_value'].sum().reset_index()
            grouped_comb_sector = group_small_slices(combined_sector_dist, 'absolute_value', 'industry_rating')
            fig7 = px.pie(grouped_comb_sector, names='industry_rating', values='absolute_value', title='Combined Sector Distribution')
            col1.plotly_chart(fig7, use_container_width=True)
            
            combined_mc_dist = total_equity_exposure.dropna(subset=['market_cap_type']).groupby('market_cap_type')['absolute_value'].sum().reset_index()
            grouped_comb_mc = group_small_slices(combined_mc_dist, 'absolute_value', 'market_cap_type')
            fig_mc3 = px.pie(grouped_comb_mc, names='market_cap_type', values='absolute_value', title='Combined Market Cap Distribution')
            col2.plotly_chart(fig_mc3, use_container_width=True)
            
            st.divider()
            st.subheader("Top 10 Combined Stock Holdings")
            top_10_combined = total_equity_exposure.sort_values('absolute_value', ascending=False).head(10)
            fig_top_combined = px.bar(top_10_combined, x='company_name', y='absolute_value', title='Top 10 Combined Stock Holdings', labels={'company_name': 'Company', 'absolute_value': 'Total Exposure Value'})
            st.plotly_chart(fig_top_combined, use_container_width=True)
            
            st.divider()
            render_track_record_table()
                
    with tab4:
        st.header("Mutual Fund Overlap Analysis")
        
        if len(st.session_state.portfolio) < 2:
            st.info("Please add at least two mutual funds to the portfolio to see the overlap analysis.")
        else:
            with st.spinner("Calculating fund overlap..."):
                overlap_df = calculate_overlap_matrix(st.session_state.portfolio, holdings_data)
                
                # Multiply non-diagonal values by 100 as requested
                display_overlap_df = overlap_df.copy()
                for i in range(len(display_overlap_df)):
                    for j in range(len(display_overlap_df)):
                        if i != j:
                            display_overlap_df.iloc[i, j] *= 100

                fig_overlap = px.imshow(
                    display_overlap_df,
                    text_auto='.1f',
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title="Portfolio Overlap Matrix"
                )
                fig_overlap.update_xaxes(side="top")
                st.plotly_chart(fig_overlap, use_container_width=True)


# --- Main App Logic ---
# This is the main dispatcher that calls the correct render function based on state.
# This structure makes the app more robust and prevents state-related errors.

if __name__ == '__main__':
    try:
        # --- Load Data ---
        engine = get_engine()
        funds_data, equities_data, holdings_data = load_all_data(engine)
        selectable_funds_data = funds_data.dropna(subset=['fund_code'])
        amc_list = selectable_funds_data['amc_name'].dropna().unique().tolist()
        equity_list = equities_data['company_name'].dropna().unique().tolist()

        # --- Session State Initialization ---
        if 'page_name' not in st.session_state:
            st.session_state.page_name = "Mutual Funds"
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        if 'stock_portfolio' not in st.session_state:
            st.session_state.stock_portfolio = []

        # --- Page Router ---
        if st.session_state.page_name == "Mutual Funds":
            render_mutual_funds_page(amc_list, selectable_funds_data)
        elif st.session_state.page_name == "Stocks":
            render_stocks_page(equity_list)
        elif st.session_state.page_name == "Analysis":
            render_analysis_page(equities_data, holdings_data)
    
    except Exception as e:
        st.error("An unexpected error occurred in the application.")
        st.exception(e)
