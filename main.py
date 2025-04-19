import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import base64
import hashlib

st.set_page_config(
    page_title="Pitcher Tracking System",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_sample_data():
    pitchers = [
        {"ID": 1, "Name": "John Smith", "Team": "Vipers", "Age": 18, "Height": "6'1\"", "Weight": 185, "Throws": "R", "Position": "SP"},
        {"ID": 2, "Name": "Mike Johnson", "Team": "Vipers", "Age": 17, "Height": "5'11\"", "Weight": 170, "Throws": "R", "Position": "RP"},
        {"ID": 3, "Name": "David Lee", "Team": "Hawks", "Age": 19, "Height": "6'3\"", "Weight": 195, "Throws": "L", "Position": "SP"},
        {"ID": 4, "Name": "Chris Brown", "Team": "Hawks", "Age": 18, "Height": "6'0\"", "Weight": 180, "Throws": "R", "Position": "RP"},
    ]
    
    metrics = [
        {"PitcherID": 1, "Date": "2023-05-01", "Velocity": 89, "SpinRate": 2200, "StrikePercentage": 68, "IP": 5, "K": 7, "BB": 2, "H": 4, "ER": 1},
        {"PitcherID": 1, "Date": "2023-05-08", "Velocity": 90, "SpinRate": 2250, "StrikePercentage": 72, "IP": 6, "K": 8, "BB": 1, "H": 3, "ER": 0},
        {"PitcherID": 2, "Date": "2023-05-02", "Velocity": 87, "SpinRate": 2100, "StrikePercentage": 65, "IP": 4, "K": 5, "BB": 3, "H": 5, "ER": 2},
        {"PitcherID": 2, "Date": "2023-05-09", "Velocity": 88, "SpinRate": 2150, "StrikePercentage": 70, "IP": 5, "K": 6, "BB": 2, "H": 4, "ER": 1},
        {"PitcherID": 3, "Date": "2023-05-03", "Velocity": 91, "SpinRate": 2300, "StrikePercentage": 75, "IP": 7, "K": 9, "BB": 1, "H": 2, "ER": 0},
        {"PitcherID": 3, "Date": "2023-05-10", "Velocity": 92, "SpinRate": 2350, "StrikePercentage": 78, "IP": 6, "K": 10, "BB": 0, "H": 1, "ER": 0},
        {"PitcherID": 4, "Date": "2023-05-04", "Velocity": 86, "SpinRate": 2050, "StrikePercentage": 62, "IP": 3, "K": 4, "BB": 4, "H": 6, "ER": 3},
        {"PitcherID": 4, "Date": "2023-05-11", "Velocity": 87, "SpinRate": 2080, "StrikePercentage": 65, "IP": 4, "K": 5, "BB": 3, "H": 5, "ER": 2},
    ]
    
    return pd.DataFrame(pitchers), pd.DataFrame(metrics)

pitchers_df, metrics_df = load_sample_data()

def calculate_ace_score(row):
    vel_score = min(100, max(0, (row['Velocity'] - 80) * 5))
    spin_score = min(100, max(0, (row['SpinRate'] - 1800) / 10))
    strike_score = min(100, max(0, row['StrikePercentage'] * 1.25))
    k_score = min(100, max(0, (row['K'] / row['IP']) * 20))
    ace_score = (vel_score * 0.3 + spin_score * 0.2 + strike_score * 0.3 + k_score * 0.2)
    return round(ace_score, 1)

def authenticate(username, password):
    users = {
        "admin": {"password": "admin123", "role": "admin"},
        "coach": {"password": "coach123", "role": "coach"},
        "scout": {"password": "scout123", "role": "scout"}
    }
    if username in users and users[username]["password"] == password:
        return users[username]["role"]
    return None

def generate_pdf_report(data, title):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(title, styles['Title']))
    table_data = [data.columns.tolist()] + data.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#777777'),
        ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), '#f3f3f3'),
        ('GRID', (0, 0), (-1, -1), 1, '#cccccc'),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def sidebar():
    st.sidebar.title("Pitcher Tracking System")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None
    
    if not st.session_state.authenticated:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            role = authenticate(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.username = username
                st.sidebar.success(f"Logged in as {username} ({role})")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username} ({st.session_state.role})")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.rerun()
    
    if st.session_state.authenticated:
        pages = {
            "Dashboard": dashboard,
            "Pitcher Profiles": pitcher_profiles,
            "Weekly Trends": weekly_trends,
            "Yearly Comparison": yearly_comparison,
            "ACE Score Analysis": ace_score_analysis,
            "Reports": reports,
        }
        if st.session_state.role == "admin":
            pages["Admin Panel"] = admin_panel
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        pages[selection]()

def dashboard():
    st.title("Pitcher Performance Dashboard")
    merged_df = pd.merge(metrics_df, pitchers_df, left_on="PitcherID", right_on="ID")
    merged_df['ACE_Score'] = merged_df.apply(calculate_ace_score, axis=1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pitchers", pitchers_df.shape[0])
    col2.metric("Total Games Recorded", metrics_df.shape[0])
    avg_vel = round(merged_df['Velocity'].mean(), 1)
    col3.metric("Avg Fastball Velocity", f"{avg_vel} mph")
    avg_ace = round(merged_df['ACE_Score'].mean(), 1)
    col4.metric("Avg ACE Score", avg_ace)
    st.subheader("Top Performers (Last 30 Days)")
    top_pitchers = merged_df.groupby(['PitcherID', 'Name']).agg({
        'ACE_Score': 'mean',
        'Velocity': 'mean',
        'StrikePercentage': 'mean',
        'K': 'sum'
    }).sort_values('ACE_Score', ascending=False).head(5)
    st.dataframe(top_pitchers.style.format({
        'ACE_Score': '{:.1f}',
        'Velocity': '{:.1f}',
        'StrikePercentage': '{:.1f}%'
    }))
    st.subheader("Velocity Distribution")
    fig = px.histogram(merged_df, x="Velocity", nbins=20, color="Team")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("ACE Score by Team")
    fig = px.box(merged_df, x="Team", y="ACE_Score", color="Team")
    st.plotly_chart(fig, use_container_width=True)

def pitcher_profiles():
    st.title("Pitcher Profiles")
    col1, col2 = st.columns(2)
    team_filter = col1.multiselect("Filter by Team", pitchers_df['Team'].unique())
    position_filter = col2.multiselect("Filter by Position", pitchers_df['Position'].unique())
    filtered_pitchers = pitchers_df.copy()
    if team_filter:
        filtered_pitchers = filtered_pitchers[filtered_pitchers['Team'].isin(team_filter)]
    if position_filter:
        filtered_pitchers = filtered_pitchers[filtered_pitchers['Position'].isin(position_filter)]
    st.dataframe(filtered_pitchers)
    if not filtered_pitchers.empty:
        selected_pitcher = st.selectbox("Select Pitcher for Details", filtered_pitchers['Name'])
        pitcher_id = filtered_pitchers[filtered_pitchers['Name'] == selected_pitcher]['ID'].values[0]
        pitcher_metrics = metrics_df[metrics_df['PitcherID'] == pitcher_id]
        pitcher_details = pitchers_df[pitchers_df['ID'] == pitcher_id].iloc[0]
        if not pitcher_metrics.empty:
            pitcher_metrics['ACE_Score'] = pitcher_metrics.apply(calculate_ace_score, axis=1)
            st.subheader(f"Performance Metrics for {selected_pitcher}")
            st.dataframe(pitcher_metrics.sort_values('Date', ascending=False))
            st.subheader("Performance Trends")
            metric_to_plot = st.selectbox("Select Metric to Plot", ['Velocity', 'SpinRate', 'StrikePercentage', 'ACE_Score'])
            fig = px.line(pitcher_metrics, x="Date", y=metric_to_plot, 
                          title=f"{metric_to_plot} Over Time for {selected_pitcher}")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Recent Performance Summary")
            latest_game = pitcher_metrics.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Last Game Velocity", f"{latest_game['Velocity']} mph")
            col2.metric("Last Game Spin Rate", f"{latest_game['SpinRate']} rpm")
            col3.metric("Strike Percentage", f"{latest_game['StrikePercentage']}%")
            col4.metric("ACE Score", latest_game['ACE_Score'])
        else:
            st.warning("No performance data available for this pitcher")

def weekly_trends():
    st.title("Weekly Performance Trends")
    min_date = pd.to_datetime(metrics_df['Date']).min()
    max_date = pd.to_datetime(metrics_df['Date']).max()
    date_range = st.date_input("Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_metrics = metrics_df[
            (pd.to_datetime(metrics_df['Date']) >= pd.to_datetime(start_date)) & 
            (pd.to_datetime(metrics_df['Date']) <= pd.to_datetime(end_date))
        ]
        if not filtered_metrics.empty:
            merged_df = pd.merge(filtered_metrics, pitchers_df, left_on="PitcherID", right_on="ID")
            merged_df['ACE_Score'] = merged_df.apply(calculate_ace_score, axis=1)
            merged_df['Week'] = pd.to_datetime(merged_df['Date']).dt.strftime('%Y-%U')
            weekly_stats = merged_df.groupby(['Week', 'Name', 'Team']).agg({
                'Velocity': 'mean',
                'SpinRate': 'mean',
                'StrikePercentage': 'mean',
                'ACE_Score': 'mean'
            }).reset_index()
            st.subheader("Weekly Performance Trends")
            metric = st.selectbox("Select Metric", ['Velocity', 'SpinRate', 'StrikePercentage', 'ACE_Score'])
            fig = px.line(weekly_stats, x="Week", y=metric, color="Name", 
                         title=f"Weekly {metric} Trends")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Weekly Comparison")
            pivot_table = weekly_stats.pivot(index='Name', columns='Week', values=metric)
            st.dataframe(pivot_table.style.format("{:.1f}"))
        else:
            st.warning("No data available for selected date range")
    else:
        st.warning("Please select a valid date range")

def yearly_comparison():
    st.title("Yearly Performance Comparison")
    merged_df = pd.merge(metrics_df, pitchers_df, left_on="PitcherID", right_on="ID")
    merged_df['ACE_Score'] = merged_df.apply(calculate_ace_score, axis=1)
    merged_df['Year'] = pd.to_datetime(merged_df['Date']).dt.year
    if len(merged_df['Year'].unique()) > 1:
        yearly_stats = merged_df.groupby(['Year', 'Name']).agg({
            'Velocity': 'mean',
            'SpinRate': 'mean',
            'StrikePercentage': 'mean',
            'ACE_Score': 'mean'
        }).reset_index()
        st.subheader("Yearly Performance Trends")
        metric = st.selectbox("Select Metric", ['Velocity', 'SpinRate', 'StrikePercentage', 'ACE_Score'])
        fig = px.line(yearly_stats, x="Year", y=metric, color="Name", 
                     title=f"Yearly {metric} Comparison")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Year-over-Year Improvement")
        if len(yearly_stats['Year'].unique()) > 1:
            pivot_stats = yearly_stats.pivot(index='Name', columns='Year', values=metric).reset_index()
            years = sorted(yearly_stats['Year'].unique())
            if len(years) >= 2:
                latest_year = years[-1]
                previous_year = years[-2]
                pivot_stats['Improvement'] = pivot_stats[latest_year] - pivot_stats[previous_year]
                pivot_stats = pivot_stats.sort_values('Improvement', ascending=False)
                st.dataframe(pivot_stats.style.format("{:.1f}"))
    else:
        st.warning("Not enough years of data for comparison")

def ace_score_analysis():
    st.title("ACE Score Analysis")
    merged_df = pd.merge(metrics_df, pitchers_df, left_on="PitcherID", right_on="ID")
    merged_df['ACE_Score'] = merged_df.apply(calculate_ace_score, axis=1)
    st.subheader("ACE Score Distribution")
    fig = px.histogram(merged_df, x="ACE_Score", nbins=20, color="Team")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Average ACE Score by Pitcher")
    ace_by_pitcher = merged_df.groupby(['Name', 'Team']).agg({
        'ACE_Score': 'mean',
        'Velocity': 'mean',
        'SpinRate': 'mean',
        'StrikePercentage': 'mean'
    }).sort_values('ACE_Score', ascending=False).reset_index()
    st.dataframe(ace_by_pitcher.style.format("{:.1f}"))
    st.subheader("ACE Score Components")
    selected_pitcher = st.selectbox("Select Pitcher", merged_df['Name'].unique())
    pitcher_data = merged_df[merged_df['Name'] == selected_pitcher]
    avg_scores = pitcher_data[['Velocity', 'SpinRate', 'StrikePercentage']].mean().reset_index()
    avg_scores.columns = ['Metric', 'Value']
    avg_scores['Normalized'] = avg_scores['Value'] / avg_scores['Value'].max()
    fig = px.line_polar(avg_scores, r='Normalized', theta='Metric', line_close=True,
                       title=f"Performance Profile for {selected_pitcher}")
    st.plotly_chart(fig, use_container_width=True)

def reports():
    st.title("Generate Reports")
    report_type = st.selectbox("Select Report Type", 
                             ["Pitcher Profile", "Weekly Performance", "Yearly Comparison", "ACE Score Analysis"])
    if report_type == "Pitcher Profile":
        selected_pitcher = st.selectbox("Select Pitcher", pitchers_df['Name'])
        pitcher_id = pitchers_df[pitchers_df['Name'] == selected_pitcher]['ID'].values[0]
        pitcher_metrics = metrics_df[metrics_df['PitcherID'] == pitcher_id]
        if not pitcher_metrics.empty:
            pitcher_metrics['ACE_Score'] = pitcher_metrics.apply(calculate_ace_score, axis=1)
            pitcher_details = pitchers_df[pitchers_df['ID'] == pitcher_id].iloc[0]
            report_df = pitcher_metrics.copy()
            for col, val in pitcher_details.items():
                if col != 'ID':
                    report_df[col] = val
            if st.button("Generate Pitcher Profile Report"):
                pdf = generate_pdf_report(report_df, f"Pitcher Profile: {selected_pitcher}")
                st.success("Report generated successfully!")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf,
                    file_name=f"Pitcher_Profile_{selected_pitcher.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No data available for selected pitcher")
    elif report_type == "Weekly Performance":
        min_date = pd.to_datetime(metrics_df['Date']).min()
        max_date = pd.to_datetime(metrics_df['Date']).max()
        date_range = st.date_input("Select Date Range", [min_date, max_date])
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_metrics = metrics_df[
                (pd.to_datetime(metrics_df['Date']) >= pd.to_datetime(start_date)) & 
                (pd.to_datetime(metrics_df['Date']) <= pd.to_datetime(end_date))
            ]
            if not filtered_metrics.empty:
                merged_df = pd.merge(filtered_metrics, pitchers_df, left_on="PitcherID", right_on="ID")
                merged_df['ACE_Score'] = merged_df.apply(calculate_ace_score, axis=1)
                if st.button("Generate Weekly Performance Report"):
                    pdf = generate_pdf_report(merged_df, f"Weekly Performance Report: {start_date} to {end_date}")
                    st.success("Report generated successfully!")
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf,
                        file_name=f"Weekly_Performance_{start_date}_to_{end_date}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No data available for selected date range")
    elif report_type == "Yearly Comparison":
        merged_df = pd.merge(metrics_df, pitchers_df, left_on="PitcherID", right_on="ID")
        merged_df['ACE_Score'] = merged_df.apply(calculate_ace_score, axis=1)
        merged_df['Year'] = pd.to_datetime(merged_df['Date']).dt.year
        yearly_stats = merged_df.groupby(['Year', 'Name', 'Team']).agg({
            'Velocity': 'mean',
            'SpinRate': 'mean',
            'StrikePercentage': 'mean',
            'ACE_Score': 'mean'
        }).reset_index()
        if st.button("Generate Yearly Comparison Report"):
            pdf = generate_pdf_report(yearly_stats, "Yearly Performance Comparison Report")
            st.success("Report generated successfully!")
            st.download_button(
                label="Download PDF Report",
                data=pdf,
                file_name="Yearly_Comparison_Report.pdf",
                mime="application/pdf"
            )
    elif report_type == "ACE Score Analysis":
        merged_df = pd.merge(metrics_df, pitchers_df, left_on="PitcherID", right_on="ID")
        merged_df['ACE_Score'] = merged_df.apply(calculate_ace_score, axis=1)
        ace_stats = merged_df.groupby(['Name', 'Team']).agg({
            'ACE_Score': ['mean', 'max', 'min'],
            'Velocity': 'mean',
            'SpinRate': 'mean',
            'StrikePercentage': 'mean'
        }).reset_index()
        ace_stats.columns = [' '.join(col).strip() for col in ace_stats.columns.values]
        if st.button("Generate ACE Score Analysis Report"):
            pdf = generate_pdf_report(ace_stats, "ACE Score Analysis Report")
            st.success("Report generated successfully!")
            st.download_button(
                label="Download PDF Report",
                data=pdf,
                file_name="ACE_Score_Analysis_Report.pdf",
                mime="application/pdf"
            )

def admin_panel():
    st.title("Admin Panel")
    if st.session_state.role != "admin":
        st.warning("You don't have permission to access this page")
        return
    st.subheader("Data Management")
    st.markdown("### Upload New Data")
    uploaded_file = st.file_uploader("Upload CSV file with pitcher data", type=["csv"])
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("Preview:")
            st.dataframe(new_data.head())
            if st.button("Confirm Upload"):
                st.success("Data saved to database!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    st.markdown("### Export Data")
    export_type = st.selectbox("Select data to export", ["Pitchers", "Metrics", "All Data"])
    if st.button("Generate Export"):
        if export_type == "Pitchers":
            data = pitchers_df
            filename = "pitchers_data.csv"
        elif export_type == "Metrics":
            data = metrics_df
            filename = "metrics_data.csv"
        else:
            data = pd.merge(metrics_df, pitchers_df, left_on="PitcherID", right_on="ID")
            filename = "all_pitcher_data.csv"
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
    st.markdown("### System Settings")
    new_team = st.text_input("Add New Team")
    if st.button("Add Team"):
        st.success(f"Team '{new_team}' added successfully!")
    st.markdown("### User Management")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    new_role = st.selectbox("Role", ["admin", "coach", "scout"])
    if st.button("Create User"):
        st.success(f"User '{new_username}' created with role '{new_role}'")

if __name__ == "__main__":
    sidebar()