import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Portfolio Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.set_option('client.showErrorDetails', True)

# --- Database Connection ---
@st.cache_resource
def get_engine():
    """Creates a SQLAlchemy engine to connect to our database."""
    db_path = 'portfolio_analysis.db' # CORRECTED DB NAME
    if not os.path.exists(db_path):
        st.error(f"Database file '{db_path}' not found. Please run the database creation scripts first.")
        st.stop()
    return create_engine(f'sqlite:///{db_path}')

# --- Data Loading Functions ---
@st.cache_data(ttl=3600)
def load_all_data(_engine):
    """Loads all necessary data from our database tables and normalizes percentage columns."""
    with _engine.connect() as connection:
        # CORRECTED TABLE NAMES
        funds_df = pd.read_sql_table('fund_amc_info', connection)
        equities_df = pd.read_sql_table('equities_info_v2', connection)
        holdings_df = pd.read_sql_table('portfolio_holdings', connection)

    # --- Normalize pct_net_assets ---
    # This is crucial because source CSVs have inconsistent formats (e.g., 5.3 vs 0.053)
    holdings_df['pct_net_assets'] = pd.to_numeric(holdings_df['pct_net_assets'], errors='coerce').fillna(0)
    
    # Heuristic: If a fund's total pct_net_assets is very low (e.g., < 5), assume it's in fractions.
    fund_pct_sums = holdings_df.groupby('fund_name')['pct_net_assets'].sum()
    funds_as_fractions = fund_pct_sums[fund_pct_sums < 5].index
    
    # Apply normalization only to funds identified as having fractional percentages
    holdings_df.loc[holdings_df['fund_name'].isin(funds_as_fractions), 'pct_net_assets'] *= 100
    
    return funds_df, equities_df, holdings_df

# --- Calculation Functions ---
def calculate_overlap_matrix(portfolio_funds, all_holdings):
    """Calculates the overlap matrix for a list of funds."""
    fund_names = [f['fund_name'] for f in portfolio_funds]
    overlap_matrix = pd.DataFrame(index=fund_names, columns=fund_names, dtype=float)
    
    equity_holdings = all_holdings[all_holdings['category'] == 'Equity'].copy()
    equity_holdings['pct_net_assets'] = pd.to_numeric(equity_holdings['pct_net_assets'], errors='coerce').fillna(0)

    for i in range(len(fund_names)):
        for j in range(i, len(fund_names)):
            fund_a_name, fund_b_name = fund_names[i], fund_names[j]
            if i == j:
                overlap_matrix.loc[fund_a_name, fund_b_name] = 100.0
                continue

            fund_a_holdings = equity_holdings[equity_holdings['fund_name'] == fund_a_name][['isin', 'pct_net_assets']]
            fund_b_holdings = equity_holdings[equity_holdings['fund_name'] == fund_b_name][['isin', 'pct_net_assets']]
            common_holdings = pd.merge(fund_a_holdings, fund_b_holdings, on='isin', suffixes=('_a', '_b'))
            
            overlap_pct = common_holdings[['pct_net_assets_a', 'pct_net_assets_b']].min(axis=1).sum()
            overlap_matrix.loc[fund_a_name, fund_b_name] = overlap_pct
            overlap_matrix.loc[fund_b_name, fund_a_name] = overlap_pct
            
    return overlap_matrix

def group_small_slices(df, value_col, name_col, threshold=3.0):
    """Groups small slices of a DataFrame into an 'Other' category for pie charts."""
    if df.empty or df[value_col].sum() <= 0: return pd.DataFrame()
    
    df['percentage'] = (df[value_col] / df[value_col].sum()) * 100
    small_slices = df[df['percentage'] < threshold]
    
    if not small_slices.empty:
        main_slices = df[df['percentage'] >= threshold]
        other_sum = small_slices[value_col].sum()
        if main_slices.empty:
            return pd.DataFrame({name_col: ['Other'], value_col: [other_sum]})
        other_row = pd.DataFrame({name_col: ['Other'], value_col: [other_sum]})
        return pd.concat([main_slices[[name_col, value_col]], other_row], ignore_index=True)
    return df[[name_col, value_col]]

# --- Page Navigation and Rendering Functions ---
def go_to_page(page):
    st.session_state.page_name = page

def reset_app():
    """Clears all portfolio data and returns to the first page."""
    st.session_state.portfolio = []
    st.session_state.stock_portfolio = []
    go_to_page("Mutual Funds")

def render_mutual_funds_page(amc_list, funds_data):
    st.title("Step 1: Build Your Mutual Fund Portfolio")
    st.write("Select a fund, enter the amount you've invested, and add it to your portfolio.")
    st.divider()
    
    col1, col2, col3, col4 = st.columns([2, 3, 1.5, 1.5])
    with col1:
        selected_amc = st.selectbox("AMC Name", options=amc_list, index=None, placeholder="Select an AMC", key="mf_amc_select")
    with col2:
        if selected_amc:
            available_funds = funds_data[funds_data['amc_name'] == selected_amc]['fund_name'].tolist()
            selected_fund = st.selectbox("Fund Name", options=available_funds, index=None, placeholder="Select a Fund", key="mf_fund_select")
        else:
            st.selectbox("Fund Name", [], disabled=True, placeholder="First select an AMC", key="mf_fund_select_disabled")
            selected_fund = None
    with col3:
        invested_value = st.number_input("Invested Value (₹)", min_value=0.0, step=1000.0, placeholder="Enter a value", key="mf_value")
    with col4:
        st.write(""); st.write("")
        add_button = st.button("Add to Portfolio", type="primary", use_container_width=True, key="mf_add_button")
    
    if add_button:
        if selected_amc and selected_fund and invested_value > 0:
            new_entry = {
                "amc_name": selected_amc, 
                "fund_name": selected_fund, 
                "value": invested_value, 
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
            row_cols[2].write(f"₹{item['value']:,.2f}")
            if row_cols[3].button("🗑️", key=f"delete_mf_{i}"):
                st.session_state.portfolio.pop(i)
                st.rerun()
        
        total_value = sum(item['value'] for item in st.session_state.portfolio)
        st.markdown(f"### Total Mutual Fund Value: **₹{total_value:,.2f}**")

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
        stock_value = st.number_input("Invested Value (₹)", min_value=0.0, step=1000.0, placeholder="Enter a value", key="stock_value")
    with col3:
        st.write(""); st.write("")
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
        for i in range(len(st.session_state.stock_portfolio) - 1, -1, -1):
            item = st.session_state.stock_portfolio[i]
            row_cols = st.columns([3, 1.5, 0.5])
            row_cols[0].write(item['company_name'])
            row_cols[1].write(f"₹{item['value']:,.2f}")
            if row_cols[2].button("🗑️", key=f"delete_stock_{i}"):
                st.session_state.stock_portfolio.pop(i)
                st.rerun()
        total_stock_value = sum(item['value'] for item in st.session_state.stock_portfolio)
        st.markdown(f"### Total Stock Value: **₹{total_stock_value:,.2f}**")
    
    st.divider()
    bcol1, bcol2, _ = st.columns([1.5, 1.5, 5])
    bcol1.button("← Back to Mutual Funds", on_click=go_to_page, args=("Mutual Funds",))
    bcol2.button("Proceed to Analysis →", on_click=go_to_page, args=("Analysis",), type="primary")

def render_analysis_page(equities_data, holdings_data):
    st.title("Step 3: Portfolio Analysis")
    bcol1, bcol2, _ = st.columns([1.5, 1.5, 5])
    bcol1.button("← Back to Stocks", on_click=go_to_page, args=("Stocks",))
    bcol2.button("Reset Portfolio", on_click=reset_app)
    st.divider()

    if not st.session_state.portfolio and not st.session_state.stock_portfolio:
        st.warning("Your portfolio is empty. Please go back and add some funds or stocks."); st.stop()

    # --- Data Processing for Analysis ---
    direct_stocks_df = pd.DataFrame(st.session_state.stock_portfolio)
    if not direct_stocks_df.empty:
        direct_stocks_df = pd.merge(direct_stocks_df, equities_data, on="company_name", how="left")

    mf_portfolio_df = pd.DataFrame(st.session_state.portfolio)
    underlying_holdings_df, mf_asset_allocation = pd.DataFrame(), pd.DataFrame()
    if not mf_portfolio_df.empty:
        merged_holdings = pd.merge(mf_portfolio_df, holdings_data, on='fund_name', how='left')
        merged_holdings['pct_net_assets'] = pd.to_numeric(merged_holdings['pct_net_assets'], errors='coerce').fillna(0)
        merged_holdings['absolute_value'] = (merged_holdings['pct_net_assets'] / 100) * merged_holdings['value']
        mf_asset_allocation = merged_holdings.groupby('category')['absolute_value'].sum().reset_index()
        underlying_holdings_df = pd.merge(merged_holdings, equities_data.drop_duplicates('isin'), on='isin', how='left', suffixes=('', '_master'))

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Stocks Only", "Mutual Funds Only", "Combined Portfolio", "Mutual Fund Overlap"])

    with tab1:
        st.header("Direct Stock Holdings Analysis")
        if direct_stocks_df.empty: st.info("You have not added any direct stocks to your portfolio.")
        else:
            positive_stocks = direct_stocks_df[direct_stocks_df['value'] > 0]
            if positive_stocks.empty: st.info("No direct stocks with positive value.")
            else:
                col1, col2 = st.columns(2)
                sector_data = positive_stocks.groupby('industry_rating')['value'].sum().reset_index()
                if not sector_data.empty and sector_data['value'].sum() > 0:
                    col1.plotly_chart(px.pie(group_small_slices(sector_data, 'value', 'industry_rating'), names='industry_rating', values='value', title='Sector Distribution'), use_container_width=True)
                else: col1.info("No positive sector data to display.")
                
                mcap_data = positive_stocks.groupby('market_cap')['value'].sum().reset_index()
                if not mcap_data.empty and mcap_data['value'].sum() > 0:
                    col2.plotly_chart(px.pie(group_small_slices(mcap_data, 'value', 'market_cap'), names='market_cap', values='value', title='Market Cap Distribution'), use_container_width=True)
                else: col2.info("No positive market cap data to display.")

                st.subheader("Top 10 Direct Stock Holdings")
                st.plotly_chart(px.bar(positive_stocks.nlargest(10, 'value'), x='company_name', y='value'), use_container_width=True)

    with tab2:
        st.header("Mutual Fund Holdings Analysis")
        if mf_portfolio_df.empty: st.info("You have not added any mutual funds.")
        else:
            col1, col2 = st.columns(2)
            amc_data = mf_portfolio_df[mf_portfolio_df['value'] > 0].groupby('amc_name')['value'].sum().reset_index()
            if not amc_data.empty:
                col1.plotly_chart(px.pie(group_small_slices(amc_data, 'value', 'amc_name'), names='amc_name', values='value', title='Distribution by AMC'), use_container_width=True)
            else: col1.info("No positive AMC data to display.")
            
            asset_data = mf_asset_allocation[mf_asset_allocation['absolute_value'] > 0]
            if not asset_data.empty:
                col2.plotly_chart(px.pie(group_small_slices(asset_data, 'absolute_value', 'category'), names='category', values='absolute_value', title='Underlying Asset Distribution'), use_container_width=True)
            else: col2.info("No positive asset data to display.")

            st.subheader("Underlying Equity Analysis")
            equity_like_cats = ['Equity', 'Derivative', 'Derivatives']
            underlying_equities = underlying_holdings_df[underlying_holdings_df['category'].isin(equity_like_cats)]
            positive_equities = underlying_equities[underlying_equities['absolute_value'] > 0]
            if positive_equities.empty: st.info("No underlying equities with positive value.")
            else:
                agg_equities = positive_equities.groupby(['company_name_master', 'industry_rating_master', 'market_cap'])['absolute_value'].sum().reset_index()
                agg_equities.rename(columns={'company_name_master': 'company_name', 'industry_rating_master': 'industry_rating'}, inplace=True)

                col3, col4 = st.columns(2)
                sector_data_mf = agg_equities.groupby('industry_rating')['absolute_value'].sum().reset_index()
                if not sector_data_mf.empty and sector_data_mf['absolute_value'].sum() > 0:
                    col3.plotly_chart(px.pie(group_small_slices(sector_data_mf, 'absolute_value', 'industry_rating'), names='industry_rating', values='absolute_value', title='Underlying Sector Distribution'), use_container_width=True)
                else: col3.info("No positive sector data.")

                mcap_data_mf = agg_equities.groupby('market_cap')['absolute_value'].sum().reset_index()
                if not mcap_data_mf.empty and mcap_data_mf['absolute_value'].sum() > 0:
                    col4.plotly_chart(px.pie(group_small_slices(mcap_data_mf, 'absolute_value', 'market_cap'), names='market_cap', values='absolute_value', title='Underlying Market Cap Distribution'), use_container_width=True)
                else: col4.info("No positive market cap data.")

                st.subheader("Top 10 Underlying Stock Holdings")
                st.plotly_chart(px.bar(agg_equities.nlargest(10, 'absolute_value'), x='company_name', y='absolute_value'), use_container_width=True)

    with tab3:
        st.header("Combined Portfolio Analysis")
        direct_stocks_for_combo = pd.DataFrame()
        if not direct_stocks_df.empty:
            direct_stocks_for_combo = direct_stocks_df[['company_name', 'industry_rating', 'market_cap', 'value']].rename(columns={'value': 'absolute_value'})
        
        mf_equities_for_combo = pd.DataFrame()
        if not underlying_holdings_df.empty:
            equity_like_cats = ['Equity', 'Derivative', 'Derivatives']
            mf_equities_for_combo = underlying_holdings_df[underlying_holdings_df['category'].isin(equity_like_cats)]
            mf_equities_for_combo = mf_equities_for_combo.groupby(['company_name_master', 'industry_rating_master', 'market_cap'])['absolute_value'].sum().reset_index()
            mf_equities_for_combo.rename(columns={'company_name_master': 'company_name', 'industry_rating_master': 'industry_rating'}, inplace=True)

        if direct_stocks_for_combo.empty and mf_equities_for_combo.empty: st.info("No equity holdings to combine.")
        else:
            combined_df = pd.concat([direct_stocks_for_combo, mf_equities_for_combo])
            total_equity_exposure = combined_df.groupby(['company_name', 'industry_rating', 'market_cap'])['absolute_value'].sum().reset_index()
            positive_combined = total_equity_exposure[total_equity_exposure['absolute_value'] > 0]

            if positive_combined.empty: st.info("No combined equities with positive value.")
            else:
                col1, col2 = st.columns(2)
                sector_data_comb = positive_combined.groupby('industry_rating')['absolute_value'].sum().reset_index()
                if not sector_data_comb.empty and sector_data_comb['absolute_value'].sum() > 0:
                    col1.plotly_chart(px.pie(group_small_slices(sector_data_comb, 'absolute_value', 'industry_rating'), names='industry_rating', values='absolute_value', title='Combined Sector Distribution'), use_container_width=True)
                else: col1.info("No positive sector data.")
                
                mcap_data_comb = positive_combined.groupby('market_cap')['absolute_value'].sum().reset_index()
                if not mcap_data_comb.empty and mcap_data_comb['absolute_value'].sum() > 0:
                    col2.plotly_chart(px.pie(group_small_slices(mcap_data_comb, 'absolute_value', 'market_cap'), names='market_cap', values='absolute_value', title='Combined Market Cap Distribution'), use_container_width=True)
                else: col2.info("No positive market cap data.")

                st.subheader("Top 10 Combined Holdings")
                st.plotly_chart(px.bar(positive_combined.nlargest(10, 'absolute_value'), x='company_name', y='absolute_value'), use_container_width=True)
                st.subheader("Full Combined Equity Holdings")
                st.dataframe(positive_combined.sort_values('absolute_value', ascending=False), use_container_width=True)
                
    with tab4:
        st.header("Mutual Fund Overlap Analysis")
        if len(st.session_state.portfolio) < 2:
            st.info("Please add at least two mutual funds to the portfolio to see the overlap analysis.")
        else:
            with st.spinner("Calculating fund overlap..."):
                overlap_df = calculate_overlap_matrix(st.session_state.portfolio, holdings_data)
                fig = px.imshow(overlap_df, text_auto='.1f', aspect="auto", color_continuous_scale='Blues', title="Portfolio Overlap Matrix (%)")
                fig.update_xaxes(side="top"); st.plotly_chart(fig, use_container_width=True)

# --- Main App Logic ---
if __name__ == '__main__':
    try:
        engine = get_engine()
        funds_data, equities_data, holdings_data = load_all_data(engine)
        amc_list = sorted(funds_data['amc_name'].dropna().unique().tolist())
        equity_list = sorted(equities_data['company_name'].dropna().unique().tolist())

        if 'page_name' not in st.session_state: st.session_state.page_name = "Mutual Funds"
        if 'portfolio' not in st.session_state: st.session_state.portfolio = []
        if 'stock_portfolio' not in st.session_state: st.session_state.stock_portfolio = []

        if st.session_state.page_name == "Mutual Funds":
            render_mutual_funds_page(amc_list, funds_data)
        elif st.session_state.page_name == "Stocks":
            render_stocks_page(equity_list)
        elif st.session_state.page_name == "Analysis":
            render_analysis_page(equities_data, holdings_data)
    
    except Exception as e:
        st.error("An unexpected error occurred in the application.")
        st.exception(e)
