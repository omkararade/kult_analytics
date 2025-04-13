import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import datetime

# Initialize NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# App Configuration
st.set_page_config(
    page_title="Josh App Analytics Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stMetric { background-color: white; border-radius: 12px; padding: 20px; 
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); border-left: 4px solid #6e48aa; }
    .alert-box { background-color: #ffebee; border-left: 4px solid #dc3545; 
                padding: 15px; margin: 10px 0; border-radius: 4px; }
    .review-card { background-color: white; border-radius: 8px; padding: 15px; 
                margin: 10px 0; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); }
</style>
""", unsafe_allow_html=True)

# Global topic keywords
topic_keywords = {
    'Delivery_Issue': r'\b(?:delivery|late|wrong order|missing item|courier|tracking)\b',
    'Performance_Issue': r'\b(?:crash|freeze|lag|slow|bug|glitch)\b',
    'Support_Complaint': r'\b(?:support|help|response|reply|assistance|service)\b',
    'UI_Issue': r'\b(?:interface|design|navigation|layout|user.?friendly)\b',
    'Payment_Issue': r'\b(?:payment|transaction|refund|charge|billing|wallet)\b',
    'Food_Quality': r'\b(?:food|taste|quality|fresh|packaging|temperature)\b',
    'Promotions': r'\b(?:coupon|discount|offer|promo|deal|voucher)\b',
    'Feature_Request': r'\b(?:need|want|please add|missing|would love|suggest|dark mode|wallet)\b',
    'Churn_Risk': r'\b(?:uninstall|stop using|never return|delete app|switch to competitor)\b'
}

# Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Swiggy_App.csv")
        if df.empty:
            st.error("Dataset is empty!")
            return pd.DataFrame()

        # Date processing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Text cleaning
        df['Review'] = df['Review'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        df['Sentiment_Score'] = df['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
        df['Sentiment'] = pd.cut(df['Sentiment_Score'], 
                               bins=[-1, -0.05, 0.05, 1], 
                               labels=['Negative', 'Neutral', 'Positive'])
        
        # Topic detection
        for topic, pattern in topic_keywords.items():
            df[topic] = df['Review'].str.contains(pattern, case=False, regex=True)
            
        return df

    except FileNotFoundError:
        st.error("File not found!")
        return pd.DataFrame()

# Metrics calculation
def compute_topic_metrics(df, topic_col):
    try:
        topic_df = df[df[topic_col]]
        if len(topic_df) == 0:
            return pd.Series()
            
        all_avg_rating = df['Rating'].mean() or 0
        return pd.Series({
            'volume': len(topic_df),
            'avg_rating': topic_df['Rating'].mean(),
            'positive_pct': (topic_df['Sentiment'] == 'Positive').mean() * 100,
            'negative_pct': (topic_df['Sentiment'] == 'Negative').mean() * 100,
            'impact': all_avg_rating - topic_df['Rating'].mean(),
            'response_priority': len(topic_df) * (topic_df['Rating'].mean() / 5)
        })
    except Exception as e:
        st.error(f"Metrics error: {str(e)}")
        return pd.Series()

# Load data
df = load_data()
if df.empty:
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['Date'].min().date(), df['Date'].max().date()),
    min_value=df['Date'].min().date(),
    max_value=df['Date'].max().date()
)

rating_range = st.sidebar.slider("Rating Range", 1, 5, (1, 5))
sentiment_filter = st.sidebar.multiselect("Sentiment", df['Sentiment'].unique(), df['Sentiment'].unique())

# Apply filters
filtered_df = df[
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Rating'].between(*rating_range)) &
    (df['Sentiment'].isin(sentiment_filter))
]

# Main Dashboard
st.title("Josh App Analytics Dashboard")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    avg_rating = filtered_df['Rating'].mean()
    st.metric("Average Rating", f"{avg_rating:.1f}")
    
with col2:
    total_reviews = len(filtered_df)
    st.metric("Total Reviews", f"{total_reviews:,}")
    
with col3:
    pos_percent = (filtered_df['Sentiment'] == 'Positive').mean() * 100
    st.metric("Positive Sentiment", f"{pos_percent:.1f}%")
    
with col4:
    neg_percent = (filtered_df['Sentiment'] == 'Negative').mean() * 100
    st.metric("Negative Sentiment", f"{neg_percent:.1f}%")

# Alerts
if avg_rating < 3.5:
    st.markdown("<div class='alert-box'><h4>🔴 ALERT: Average rating below 3.5!</h4></div>", unsafe_allow_html=True)
if neg_percent > 30:
    st.markdown("<div class='alert-box'><h4>🔴 ALERT: High negative sentiment!</h4></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📝 Reviews", "🔍 Insights", "⚠️ Issues", "📄 Report"])

with tab1:  # Trends Tab
    st.header("Temporal Analysis")
    freq = st.radio("Frequency:", ["Weekly", "Monthly", "Quarterly"], horizontal=True)
    
    if freq == "Weekly":
        filtered_df['Period'] = filtered_df['Date'].dt.strftime('%Y-%U')
    elif freq == "Monthly":
        filtered_df['Period'] = filtered_df['Date'].dt.strftime('%Y-%m')
    else:
        filtered_df['Period'] = filtered_df['Date'].dt.to_period('Q').astype(str)

    # Sentiment trend
        # In Temporal Analysis tab
    sentiment_counts = filtered_df.groupby(
        ['Period', 'Sentiment'], observed=False
    ).size().unstack(fill_value=0)
    fig = px.area(sentiment_counts, title="Sentiment Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Rating trend
    rating_trend = filtered_df.groupby('Period')['Rating'].mean().reset_index()
    fig = px.line(rating_trend, x='Period', y='Rating', title="Rating Trend")
    st.plotly_chart(fig, use_container_width=True)

with tab2:  # Reviews Tab
    st.header("Review Analysis")
    search_term = st.text_input("Search reviews:")
    results = filtered_df[filtered_df['Review'].str.contains(search_term, case=False)] if search_term else filtered_df
    
    sample_size = st.slider("Sample Size", 5, 100, 20)
    for _, row in results.sample(min(sample_size, len(results))).iterrows():
        sentiment_color = "#d4edda" if row['Sentiment'] == 'Positive' else "#fff3cd" if row['Sentiment'] == 'Neutral' else "#f8d7da"
        st.markdown(f"""
        <div class="review-card" style="border-left: 4px solid {sentiment_color};">
            <div style="display: flex; justify-content: space-between;">
                <div>⭐ {row['Rating']}/5</div>
                <div>{row['Date'].strftime('%Y-%m-%d')}</div>
            </div>
            <p>{row['Review']}</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:  # Insights Tab
    st.header("Key Insights")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.pie(filtered_df, names='Sentiment', hole=0.4, 
                    title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Topic Sentiment Analysis")
        topic_data = []
        for topic in topic_keywords:
            topic_df = filtered_df[filtered_df[topic]]
            if not topic_df.empty:
                topic_data.append({
                    'Topic': topic.replace('_', ' '),
                    'Positive%': (topic_df['Sentiment'] == 'Positive').mean() * 100,
                    'Negative%': (topic_df['Sentiment'] == 'Negative').mean() * 100,
                    'Avg Rating': topic_df['Rating'].mean()
                })
        st.dataframe(pd.DataFrame(topic_data), height=300)

with tab4:  # Issues Tab
    st.header("Issue Prioritization")
    
    # Urgency Heatmap
    topic_metrics = pd.DataFrame({topic: compute_topic_metrics(filtered_df, topic) 
                                for topic in topic_keywords}).T.reset_index()
    fig = px.scatter(topic_metrics, x='volume', y='impact', size='response_priority',
                    hover_name='index', log_x=True, title="Issue Priority Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Requests
    st.subheader("Feature Requests Analysis")
    feature_requests = filtered_df[filtered_df['Feature_Request']]
    if not feature_requests.empty:
        features = Counter()
        for review in feature_requests['Review']:
            blob = TextBlob(review)
            features.update([np for np in blob.noun_phrases if any(kw in np for kw in ['app', 'feature', 'mode', 'track'])])
        
        if features:
            features_df = pd.DataFrame(
                features.most_common(10), 
                columns=['Feature', 'Mentions']
            )
            fig = px.bar(
                features_df,
                x='Mentions',
                y='Feature',
                orientation='h',
                title="Top Requested Features"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Churn Risk
    st.subheader("Churn Risk Detection")
    churn_reviews = filtered_df[filtered_df['Churn_Risk']]
    if not churn_reviews.empty:
        for _, row in churn_reviews.iterrows():
            st.markdown(f"**⭐{row['Rating']}** - {row['Review']}")
            st.markdown("*Suggested response*: We apologize for your experience...")
with tab5:  # Report Tab
    st.header("Download Report")
    report_content = f"""
    Josh App Analytics Report
    Period: {date_range[0]} to {date_range[1]}
    
    Key Metrics:
    - Avg Rating: {avg_rating:.1f}
    - Total Reviews: {total_reviews:,}
    - Positive Sentiment: {pos_percent:.1f}%
    - Negative Sentiment: {neg_percent:.1f}%
    
    Recommendations:
    1. Address high-impact issues from priority matrix
    2. Improve response time for support tickets
    3. Prioritize top feature requests
    """
    st.download_button("Download Report", report_content, "josh_report.txt")

# Sidebar Info
st.sidebar.markdown("""
---
**Dashboard Guide**
1. Use date and rating filters
2. Explore different tabs
3. Click charts for details
4. Download reports from Report tab
""")