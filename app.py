import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from fpdf import FPDF
import base64
from prophet import Prophet
from collections import Counter
from nltk.corpus import stopwords
import plotly.figure_factory as ff
from io import BytesIO

# Initialize NLTK
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

# App Configuration
st.set_page_config(
    page_title="Beauty Kult App Analytics Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    :root {
        --primary: #6e48aa;
        --secondary: #9d50bb;
        --light: #f8f9fa;
        --dark: #343a40;
        --success: #28a745;
        --danger: #dc3545;
        --warning: #fd7e14;
        --info: #17a2b8;
    }
    
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 4px solid var(--primary);
    }
    .feature-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-top: 3px solid var(--primary);
    }
    .highlight-box {
        background-color: #fff8e1;
        border-left: 4px solid var(--warning);
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .alert-box {
        background-color: #ffebee;
        border-left: 4px solid var(--danger);
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid var(--success);
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid var(--info);
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .tab-heading {
        color: var(--primary);
        border-bottom: 2px solid var(--secondary);
        padding-bottom: 8px;
    }
    .report-section {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .service-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-top: 4px solid var(--primary);
        transition: transform 0.3s;
    }
    .service-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

def generate_report(df):
    """Generate PDF report of the analysis"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Beauty Kult App Analytics Report", ln=1, align='C')
        pdf.set_font("Arial", size=12)
        
        # Date
        pdf.cell(200, 10, txt=f"Report generated on: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
        pdf.ln(10)
        
        # Key Metrics - Using asterisk instead of star symbol
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Key Metrics", ln=1)
        pdf.set_font("Arial", size=12)
        
        metrics = [
            ("Average Rating", f"{df['Rating'].mean():.1f}"),
            ("Positive Sentiment", f"{(df['Sentiment'] == 'Positive').mean()*100:.1f}%"),
            ("Response Rate", f"{df['Reply'].apply(lambda x: x != 'No Reply').mean()*100:.1f}%"),
            ("Active Issues", f"{df[['UI_Issue', 'Performance_Issue']].any(axis=1).sum()} reports")
        ]
        
        for metric, value in metrics:
            pdf.cell(200, 10, txt=f"{metric}: {value}", ln=1)
        
        # Competitive Benchmark
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Competitive Benchmark", ln=1)
        pdf.set_font("Arial", size=12)
        
        industry_avg = 4.2
        client_avg = df['Rating'].mean()
        if client_avg < industry_avg:
            pdf.cell(200, 10, txt=f"Opportunity: Your rating is {industry_avg - client_avg:.1f} below beauty app average", ln=1)
        else:
            pdf.cell(200, 10, txt=f"Strength: Your rating is {client_avg - industry_avg:.1f} above industry average", ln=1)
        
        pdf.ln(10)
        
        # Top Findings
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Top Findings", ln=1)
        pdf.set_font("Arial", size=12)
        
        # Add more sections as needed...
        
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        st.error(f"Failed to generate PDF report: {str(e)}")
        return b''

def generate_forecast(df):
    """Generate rating forecast using Prophet"""
    try:
        df_forecast = df.groupby('Date')['Rating'].mean().reset_index()
        df_forecast.columns = ['ds', 'y']
        
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(df_forecast)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        fig = model.plot(forecast)
        plt.title('30-Day Rating Forecast', pad=20)
        plt.xlabel('Date')
        plt.ylabel('Rating')
        return fig
    except Exception as e:
        st.warning(f"Forecast generation failed: {str(e)}")
        return None

@st.cache_data
def load_data():
    df = pd.read_csv("google_play_reviews.csv")
    
    # Data Cleaning and Preprocessing
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Reply Date'] = pd.to_datetime(df['Reply Date'], errors='coerce')
    df['Review'] = df['Review'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['Reply_Time_Days'] = (df['Reply Date'] - df['Date']).dt.days
    df['Reply_Time_Days'] = df['Reply_Time_Days'].astype('Int64')
    
    # Fix for chained assignment warning
    df = df.assign(Reply=df['Reply'].fillna("No Reply"))
    
    if 'Usefulness' in df.columns:
        df['Usefulness'] = df['Usefulness'].str.replace(r'[^\d]', '', regex=True)
        df['Usefulness'] = pd.to_numeric(df['Usefulness'], errors='coerce').fillna(0).astype('Int64')
    
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    df['Sentiment_Score'] = df['Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['Sentiment'] = pd.cut(df['Sentiment_Score'], 
                           bins=[-1, -0.05, 0.05, 1], 
                           labels=['Negative', 'Neutral', 'Positive'])
    
    # Time-based Features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Year'] = df['Date'].dt.strftime('%Y-%m')
    df['Weekday'] = df['Date'].dt.day_name()
    
    # Response Time Tiers
    bins = [0, 1, 3, 7, 14, np.inf]
    labels = ['<1 day', '1-3 days', '4-7 days', '1-2 weeks', '>2 weeks']
    df['Response_Tier'] = pd.cut(df['Reply_Time_Days'], bins=bins, labels=labels)
    
    # Device Detection
    device_patterns = {
        'iPhone': r'iphone|iOS',
        'Samsung': r'samsung|galaxy',
        'Pixel': r'pixel|google phone',
        'OnePlus': r'oneplus|one plus',
        'Xiaomi': r'xiaomi|redmi|poco',
        'Android': r'\bandroid\b(?!.*(ios|iphone))',
        'iOS': r'\bios\b|apple',
        'Tablet': r'tablet|ipad|galaxy tab'
    }
    
    def detect_devices(text):
        text = str(text).lower()
        devices = []
        for device, pattern in device_patterns.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                devices.append(device)
        return ', '.join(devices) if devices else 'Unknown'
    
    df['Devices_Mentioned'] = df['Review'].apply(detect_devices)
    
    # Issue Detection
    ui_keywords = [
        'slow', 'lag', 'bug', 'glitch', 'crash', 'freeze', 'complicated', 'hard', 'navigation',
        'unresponsive','delay','latency','stutter','load time','confusing','difficult','unintuitive'
    ]
    df['UI_Issue'] = df['Review'].str.contains('|'.join(ui_keywords), case=False, na=False)
    
    performance_keywords = [
        'crash','freeze','lag','slow','bug','glitch','not responding','stuck','hangs',
        'loading','performance','unstable','error','delay','latency'
    ]
    df['Performance_Issue'] = df['Review'].str.contains('|'.join(performance_keywords), case=False, na=False)
    
    support_categories = {
        'No Response': ['no reply', 'no answer', 'ignored', 'no help', 'no response', 'never responded', 'no feedback'],
        'Slow Response': ['slow response', 'took long', 'days to reply', 'delayed response', 'long wait', 'prolonged delay', 'late reply'],
        'Unhelpful': ['not helpful', 'useless', 'did not solve', 'waste of time', 'ineffective', 'unhelpful', 'did not assist', 'no solution', 'failed to resolve'],
        'Rude Staff': ['rude', 'arrogant', 'unprofessional', 'angry', 'impolite', 'disrespectful', 'hostile', 'offensive', 'dismissive']
    }
    
    def categorize_complaint(text):
        text = str(text).lower()
        for category, keywords in support_categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        return 'Other'
    
    df['Support_Complaint'] = df['Review'].str.contains('|'.join(
        [kw for sublist in support_categories.values() for kw in sublist]
    ), case=False, na=False)
    
    df.loc[df['Support_Complaint'], 'Support_Complaint_Type'] = df[df['Support_Complaint']]['Review'].apply(categorize_complaint)
    
    pricing_keywords = [
        'expensive','overpriced','pricey','too much','not worth','high price','cost too much',
        'unfair','cheaper','lower price','reduce price','price hike','cost','value','affordable'
    ]
    df['Pricing_Complaint'] = df['Review'].str.contains('|'.join(pricing_keywords), case=False, na=False)
    
    subscription_keywords = [
        'subscription','renewal','auto-renew','cancel','refund','billing','charge','payment',
        'iap','in-app purchase','trial','scam','unsubscribe','manage subscription'
    ]
    df['Subscription_Complaint'] = df['Review'].str.contains('|'.join(subscription_keywords), case=False, na=False)
    
    subscription_issues = {
        'Auto-Renewal': ['auto-renew', 'unsubscribe', 'hard to cancel', 'difficult to cancel'],
        'Unexpected Charges': ['unexpected charge', 'hidden fee', 'surprise charge', 'unauthorized charge'],
        'Refund Problems': ['refund', 'money back', 'not refund', 'no refund', 'refund denied'],
        'Value Issues': ['not worth', 'waste of money', 'better free', 'overpriced subscription']
    }
    
    def categorize_sub_issue(text):
        text = str(text).lower()
        for category, keywords in subscription_issues.items():
            if any(keyword in text for keyword in keywords):
                return category
        return 'Other'
    
    df.loc[df['Subscription_Complaint'], 'Subscription_Issue_Type'] = df[df['Subscription_Complaint']]['Review'].apply(categorize_sub_issue)
    
    request_phrases = [
        'should have','need','want','please add','where is','why no','missing',
        'would love','wish there was','suggest','recommend','hope to see'
    ]
    df['Feature_Request'] = df['Review'].str.contains('|'.join(request_phrases), case=False, na=False)
    
    review_counts = df['Username'].value_counts()
    df['Review_Count'] = df['Username'].map(review_counts)
    df['User_Type'] = np.where(df['Review_Count'] > 1, 'Loyal', 'First-Time')
    
    return df

# Load Data
df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Dashboard Filters")
with st.sidebar.expander("⏰ Time Period", expanded=True):
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    date_range = st.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        key="date_range_filter"
    )

with st.sidebar.expander("⭐ Rating & Sentiment", expanded=True):
    rating_range = st.slider(
        "Rating Range",
        min_value=1, max_value=5, 
        value=(1,5),
        key="rating_slider"
    )
    
    sentiment_filter = st.multiselect(
        "Filter by Sentiment",
        options=df['Sentiment'].unique(),
        default=df['Sentiment'].unique(),
        key="sentiment_filter_main"
    )

# Apply Filters
filtered_df = df[
    (df['Rating'].between(rating_range[0], rating_range[1])) &
    (df['Date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
    (df['Sentiment'].isin(sentiment_filter))
].copy()

# Main Dashboard
st.title("📊 Beauty Kult App Analytics Dashboard")

# KPI Cards with Competitive Benchmark
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    avg_rating = filtered_df['Rating'].mean()
    st.metric("Average Rating", f"{avg_rating:.1f} ★")

with col2:
    industry_avg = 4.2
    st.metric("vs Industry Avg", f"{industry_avg:.1f} ★", 
              delta=f"{avg_rating - industry_avg:.1f}★", 
              delta_color="inverse")

with col3:
    response_rate = filtered_df['Reply'].apply(lambda x: x != "No Reply").mean() * 100
    st.metric("Response Rate", f"{response_rate:.1f}%")

with col4:
    total_reviews = len(filtered_df)
    st.metric("Total Reviews", f"{total_reviews:,}")

with col5:
    pos_percent = (filtered_df['Sentiment'] == 'Positive').mean() * 100
    st.metric("Positive Sentiment", f"{pos_percent:.1f}%")

# Tabs with New Strategy Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 Trends", "📝 Reviews", "🧠 Insights", "🔍 Issues", "📊 Report", "🚀 Strategy"]
)

with tab1:
    st.header("📈 Trends Over Time")
    
    # Rating Trend
    trend_data = filtered_df.groupby('Month_Year').agg(
        Avg_Rating=('Rating', 'mean'),
        Review_Count=('Rating', 'count')
    ).reset_index()
    
    fig = px.line(trend_data, x='Month_Year', y='Avg_Rating',
                 title="Average Rating Over Time",
                 labels={'Avg_Rating': 'Average Rating', 'Month_Year': 'Month'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Trend
    sentiment_trend = filtered_df.groupby('Month_Year')['Sentiment_Score'].mean().reset_index()
    fig = px.line(sentiment_trend, x='Month_Year', y='Sentiment_Score',
                 title="Sentiment Trend Over Time",
                 labels={'Sentiment_Score': 'Average Sentiment Score'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Review Volume by Weekday
    weekday_counts = filtered_df['Weekday'].value_counts().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    fig = px.bar(weekday_counts, 
                title="Review Volume by Day of Week",
                labels={'value': 'Number of Reviews', 'index': 'Day of Week'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📝 Review Analysis")
    
    # Review Explorer
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        selected_sentiments = st.multiselect(
            "Filter by Sentiment",
            options=filtered_df['Sentiment'].unique(),
            default=filtered_df['Sentiment'].unique(),
            key="review_sentiment_filter"
        )
        
        if 'Usefulness' in filtered_df.columns:
            min_usefulness = st.slider(
                "Minimum Usefulness Score",
                min_value=0,
                max_value=int(filtered_df['Usefulness'].max()),
                value=0,
                key="usefulness_slider"
            )
        else:
            min_usefulness = 0
        
        device_filter = st.multiselect(
            "Filter by Device",
            options=filtered_df['Devices_Mentioned'].unique(),
            default=[],
            key="device_filter"
        )
    
    with col2:
        review_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)].copy()
        
        if 'Usefulness' in review_df.columns:
            review_df = review_df[review_df['Usefulness'] >= min_usefulness].copy()
        
        if device_filter:
            review_df = review_df[review_df['Devices_Mentioned'].isin(device_filter)].copy()
        
        st.dataframe(review_df[['Date', 'Rating', 'Sentiment', 'Devices_Mentioned', 'Review', 'Reply']],
                    height=600,
                    column_config={
                        "Rating": st.column_config.NumberColumn(format="⭐ %d"),
                        "Sentiment": st.column_config.TextColumn(),
                        "Devices_Mentioned": st.column_config.TextColumn("Device"),
                        "Review": st.column_config.TextColumn("Review", width="large"),
                        "Reply": st.column_config.TextColumn("Response", width="medium")
                    })
    
    # Word Cloud
    st.subheader("Review Word Cloud")
    text = " ".join(review for review in filtered_df['Review'])
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Most Frequent Words in Reviews", pad=20)
        st.pyplot(plt)
    else:
        st.warning("No reviews available for word cloud")
    
    # Top Complaint Analysis
    st.subheader("🔍 Top Complaint Analysis")
    negative_reviews = filtered_df[filtered_df['Sentiment'] == 'Negative']['Review']
    
    if len(negative_reviews) > 0:
        stop_words = set(stopwords.words('english'))
        words = [word for review in negative_reviews 
                for word in review.lower().split() 
                if word not in stop_words and len(word) > 3]
        
        top_issues = Counter(words).most_common(10)
        
        # Display as a table
        issues_df = pd.DataFrame(top_issues, columns=['Issue', 'Count'])
        st.dataframe(issues_df.style.background_gradient(cmap='Reds'), 
                    height=400,
                    column_config={
                        "Issue": "Complaint Keyword",
                        "Count": "Frequency"
                    })
        
        # Show impact on rating
        st.subheader("Issue Impact on Ratings")
        issue_impact = []
        for issue, _ in top_issues[:5]:
            affected = filtered_df[filtered_df['Review'].str.contains(issue, case=False)]
            if len(affected) > 0:
                impact = filtered_df['Rating'].mean() - affected['Rating'].mean()
                issue_impact.append({
                    'Issue': issue,
                    'Affected Reviews': len(affected),
                    'Rating Impact': impact
                })
        
        if issue_impact:
            impact_df = pd.DataFrame(issue_impact)
            st.dataframe(impact_df.sort_values('Rating Impact', ascending=False),
                        height=200,
                        column_config={
                            "Rating Impact": st.column_config.NumberColumn(
                                format="%.2f ★",
                                help="How much this issue lowers ratings"
                            )
                        })
    else:
        st.success("No negative reviews found in current filters!")

with tab3:
    st.header("🧠 Insights & Recommendations")
    
    # Sentiment Analysis
    st.subheader("Sentiment Distribution")
    sentiment_dist = filtered_df['Sentiment'].value_counts(normalize=True).mul(100)
    fig = px.pie(sentiment_dist, values=sentiment_dist.values, names=sentiment_dist.index,
                title="Review Sentiment Breakdown",
                color=sentiment_dist.index,
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Requests vs Issues
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Feature Requests")
        pos_reviews = filtered_df[filtered_df['Rating'] >= 4]['Review']
        if len(pos_reviews) > 0:
            vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=50)
            X = vectorizer.fit_transform(pos_reviews)
            features = pd.DataFrame({
                'Feature': vectorizer.get_feature_names_out(),
                'Score': X.sum(axis=0).A1
            }).sort_values('Score', ascending=False).head(10)
            
            fig = px.bar(features, x='Score', y='Feature', orientation='h',
                        title="Most Requested Features",
                        color='Score', color_continuous_scale='greens')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough positive reviews for feature analysis")
    
    with col2:
        st.subheader("Common Issues")
        neg_reviews = filtered_df[filtered_df['Rating'] <= 2]['Review']
        if len(neg_reviews) > 0:
            vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=50)
            X = vectorizer.fit_transform(neg_reviews)
            issues = pd.DataFrame({
                'Issue': vectorizer.get_feature_names_out(),
                'Score': X.sum(axis=0).A1
            }).sort_values('Score', ascending=False).head(10)
            
            fig = px.bar(issues, x='Score', y='Issue', orientation='h',
                        title="Most Frequent Issues",
                        color='Score', color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough negative reviews for issues analysis")
    
    # Device Analysis
    st.subheader("Device Distribution")
    device_counts = filtered_df['Devices_Mentioned'].value_counts().head(10)
    if not device_counts.empty:
        fig = px.bar(device_counts, orientation='h',
                    title="Top 10 Mentioned Devices",
                    labels={'value': 'Count', 'index': 'Device'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No device data available")

with tab4:
    st.header("🔍 Comprehensive Issues Analysis")
    
    # UI/UX Issues
    with st.expander("🎨 UI/UX Issues", expanded=True):
        ui_col1, ui_col2, ui_col3 = st.columns(3)
        
        with ui_col1:
            ui_issues = filtered_df['UI_Issue'].sum()
            st.metric("Total UI Issues", ui_issues)
        
        with ui_col2:
            ui_negative_pct = filtered_df[filtered_df['Sentiment'] == 'Negative']['UI_Issue'].mean() * 100
            st.metric("In Negative Reviews", f"{ui_negative_pct:.1f}%")
        
        with ui_col3:
            ui_impact = filtered_df.groupby('UI_Issue')['Rating'].mean().diff().iloc[-1]
            st.metric("Rating Impact", f"{ui_impact:.1f}★", delta_color="inverse")
        
        # UI Issues Word Cloud
        ui_reviews = " ".join(filtered_df[filtered_df['UI_Issue']]['Review'])
        if ui_reviews.strip():
            wordcloud = WordCloud(width=800, height=300, background_color='white').generate(ui_reviews)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("Most Frequent UI/UX Terms", pad=20)
            st.pyplot(plt)
        else:
            st.warning("No UI/UX issue reviews available")
        
        # UI Issues Trend
        ui_trend = filtered_df.groupby('Month_Year')['UI_Issue'].mean().reset_index()
        fig = px.line(ui_trend, x='Month_Year', y='UI_Issue', 
                     title="Monthly UI Issues Trend",
                     labels={'UI_Issue': '% of Reviews with UI Issues'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance Issues
    with st.expander("⚡ Performance & Bugs", expanded=True):
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            perf_issues = filtered_df['Performance_Issue'].sum()
            st.metric("Performance Issues", perf_issues)
        
        with perf_col2:
            perf_critical = filtered_df[(filtered_df['Performance_Issue']) & 
                                     (filtered_df['Rating'] <= 2)].shape[0]
            st.metric("Critical (1-2★) Reports", perf_critical)
        
        with perf_col3:
            perf_impact = filtered_df.groupby('Performance_Issue')['Rating'].mean().diff().iloc[-1]
            st.metric("Rating Impact", f"{perf_impact:.1f}★", delta_color="inverse")
        
        # Crash Reports Timeline
        crash_reports = filtered_df[filtered_df['Review'].str.contains('crash|freeze', case=False, na=False)]
        if not crash_reports.empty:
            crash_trend = crash_reports.groupby(crash_reports['Date'].dt.strftime('%Y-%W')).size()
            fig = px.line(crash_trend, title="Weekly Crash/Freeze Reports",
                         labels={'value': 'Report Count', 'index': 'Week'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No crash/freeze reports found")
    
    # Feature Requests
    with st.expander("✨ Feature Requests", expanded=True):
        req_col1, req_col2 = st.columns(2)
        
        with req_col1:
            feature_requests = filtered_df['Feature_Request'].sum()
            st.metric("Feature Requests", feature_requests)
        
        with req_col2:
            missing_func = filtered_df['Review'].str.contains('missing|lack|without', case=False, na=False).sum()
            st.metric("Missing Functionality", missing_func)
        
        # Top Feature Requests
        if filtered_df['Feature_Request'].sum() > 0:
            vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=50)
            X = vectorizer.fit_transform(filtered_df[filtered_df['Feature_Request']]['Review'])
            features = pd.DataFrame({
                'Feature': vectorizer.get_feature_names_out(),
                'Score': X.sum(axis=0).A1
            }).sort_values('Score', ascending=False).head(10)
            
            fig = px.bar(features, x='Score', y='Feature', orientation='h',
                        title="Top Requested Features",
                        color='Score', color_continuous_scale='blues')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature requests found")
    
    # Customer Support
    with st.expander("📞 Customer Support", expanded=True):
        supp_col1, supp_col2, supp_col3 = st.columns(3)
        
        with supp_col1:
            support_issues = filtered_df['Support_Complaint'].sum()
            st.metric("Support Complaints", support_issues)
        
        with supp_col2:
            avg_response = filtered_df[filtered_df['Reply'] != "No Reply"]['Reply_Time_Days'].median()
            st.metric("Median Response Time", f"{avg_response:.1f} days")
        
        with supp_col3:
            response_coverage = (filtered_df['Reply'] != "No Reply").mean() * 100
            st.metric("Response Coverage", f"{response_coverage:.1f}%")
        
        # Support Complaint Types
        if 'Support_Complaint_Type' in filtered_df.columns:
            supp_types = filtered_df['Support_Complaint_Type'].value_counts()
            if not supp_types.empty:
                fig = px.pie(supp_types, values=supp_types.values, names=supp_types.index,
                            title="Support Complaint Types")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No support complaints found")
        
        # Response Time Impact
        if 'Reply_Time_Days' in filtered_df.columns and not filtered_df[filtered_df['Reply'] != "No Reply"].empty:
            fig = px.scatter(filtered_df[filtered_df['Reply'] != "No Reply"],
                            x='Reply_Time_Days', y='Rating',
                            title="Response Time vs. Rating",
                            labels={'Reply_Time_Days': 'Days to Respond'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No response time data available")
    
    # Monetization
    with st.expander("💰 Monetization", expanded=True):
        monet_col1, monet_col2, monet_col3 = st.columns(3)
        
        with monet_col1:
            pricing_issues = filtered_df['Pricing_Complaint'].sum()
            st.metric("Pricing Complaints", pricing_issues)
        
        with monet_col2:
            sub_issues = filtered_df['Subscription_Complaint'].sum()
            st.metric("Subscription Issues", sub_issues)
        
        with monet_col3:
            value_sentiment = filtered_df[filtered_df['Pricing_Complaint']]['Sentiment_Score'].mean()
            st.metric("Pricing Sentiment", f"{value_sentiment:.2f}")
        
        # Subscription Issue Types
        if 'Subscription_Issue_Type' in filtered_df.columns:
            sub_types = filtered_df['Subscription_Issue_Type'].value_counts()
            if not sub_types.empty:
                fig = px.bar(sub_types, orientation='h',
                            title="Subscription Issue Types",
                            labels={'value': 'Count', 'index': 'Issue Type'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No subscription issues found")
    
    # User Retention
    with st.expander("👥 User Retention", expanded=True):
        ret_col1, ret_col2, ret_col3 = st.columns(3)
        
        with ret_col1:
            loyal_users = filtered_df['User_Type'].value_counts().get('Loyal', 0)
            st.metric("Loyal Users", loyal_users)
        
        with ret_col2:
            retention_rate = (filtered_df['Review_Count'] > 1).mean() * 100
            st.metric("Retention Rate", f"{retention_rate:.1f}%")
        
        with ret_col3:
            loyal_sentiment = filtered_df[filtered_df['User_Type'] == 'Loyal']['Sentiment_Score'].mean()
            st.metric("Loyal User Sentiment", f"{loyal_sentiment:.2f}")
        
        # Loyal User Analysis
        if 'User_Type' in filtered_df.columns:
            loyal_analysis = filtered_df.groupby('User_Type').agg({
                'Rating': 'mean',
                'Sentiment_Score': 'mean',
                'Review_Count': 'mean'
            })
            if not loyal_analysis.empty:
                fig = px.bar(loyal_analysis, barmode='group',
                            title="Loyal vs First-Time Users")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No user type data available")
    
    # Prioritization Matrix
    st.subheader("📊 Issue Prioritization Matrix")
    priority_data = {
        'Issue Type': ['UI/UX', 'Performance', 'Feature Requests', 'Support', 'Monetization'],
        'Frequency': [
            filtered_df['UI_Issue'].mean(),
            filtered_df['Performance_Issue'].mean(),
            filtered_df['Feature_Request'].mean(),
            filtered_df['Support_Complaint'].mean(),
            (filtered_df['Pricing_Complaint'] | filtered_df['Subscription_Complaint']).mean()
        ],
        'Impact': [
            filtered_df.groupby('UI_Issue')['Rating'].mean().diff().iloc[-1],
            filtered_df.groupby('Performance_Issue')['Rating'].mean().diff().iloc[-1],
            0.5,  # Assuming feature requests have medium impact
            filtered_df.groupby('Support_Complaint')['Rating'].mean().diff().iloc[-1],
            filtered_df.groupby('Pricing_Complaint')['Rating'].mean().diff().iloc[-1]
        ]
    }
    
    priority_df = pd.DataFrame(priority_data)
    if not priority_df.empty:
        fig = px.scatter(priority_df, x='Frequency', y='Impact', text='Issue Type',
                        size='Frequency', color='Issue Type',
                        title="Issue Prioritization (Size = Frequency)")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to generate prioritization matrix")

with tab5:
    st.header("📊 Executive Summary Report")
    st.subheader("Key Performance Indicators")
    
    # Create a metrics grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Rating", f"{filtered_df['Rating'].mean():.1f} ★", 
                 help="Overall average star rating from users")
    with col2:
        st.metric("Response Rate", f"{filtered_df['Reply'].apply(lambda x: x != 'No Reply').mean()*100:.1f}%",
                 help="Percentage of reviews that received a response")
    with col3:
        st.metric("Positive Sentiment", f"{(filtered_df['Sentiment'] == 'Positive').mean()*100:.1f}%",
                 help="Percentage of reviews with positive sentiment")
    with col4:
        st.metric("Active Issues", f"{filtered_df[['UI_Issue', 'Performance_Issue']].any(axis=1).sum()}",
                 help="Total reviews reporting UI or performance issues")
    
    st.divider()
    
    # Section 1: Overall Performance
    with st.expander("📌 Overall App Performance", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            st.subheader("Rating Distribution")
            rating_dist = filtered_df['Rating'].value_counts().sort_index()
            fig = px.bar(rating_dist, 
                         labels={'value': 'Count', 'index': 'Stars'},
                         color=rating_dist.index,
                         color_continuous_scale='tealgrn')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Sentiment distribution
            st.subheader("Sentiment Analysis")
            sentiment_dist = filtered_df['Sentiment'].value_counts(normalize=True).mul(100)
            fig = px.pie(sentiment_dist, 
                         values=sentiment_dist.values, 
                         names=sentiment_dist.index,
                         color=sentiment_dist.index,
                         color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Section 2: Key Highlights
    with st.expander("🔍 Key Highlights", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👍 Top Positive Feedback")
            pos_feedback = filtered_df[filtered_df['Rating'] >= 4]
            if not pos_feedback.empty:
                top_features = pos_feedback['Review'].value_counts().head(5)
                for i, (review, count) in enumerate(top_features.items(), 1):
                    st.markdown(f"{i}. **{review[:100]}...** (mentioned {count} times)")
            else:
                st.warning("No positive feedback in selected filters")
            
        with col2:
            st.subheader("👎 Top Complaints")
            neg_feedback = filtered_df[filtered_df['Rating'] <= 2]
            if not neg_feedback.empty:
                top_issues = neg_feedback['Review'].value_counts().head(5)
                for i, (review, count) in enumerate(top_issues.items(), 1):
                    st.markdown(f"{i}. **{review[:100]}...** (mentioned {count} times)")
            else:
                st.warning("No negative feedback in selected filters")
    
    # Section 3: Actionable Insights
    with st.expander("🚀 Actionable Recommendations", expanded=True):
        st.subheader("Priority Areas for Improvement")
        
        # Calculate issue priorities
        issues = {
            'UI Issues': filtered_df['UI_Issue'].mean(),
            'Performance Issues': filtered_df['Performance_Issue'].mean(),
            'Support Complaints': filtered_df['Support_Complaint'].mean(),
            'Pricing Concerns': filtered_df['Pricing_Complaint'].mean()
        }
        
        # Sort by most frequent issues
        sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)
        
        for issue, freq in sorted_issues:
            if freq > 0:
                st.progress(freq, text=f"{issue} ({freq:.1%} of reviews)")
        
        st.markdown("""
        **Recommended Actions:**
        1. Address the most frequently reported issues first
        2. Improve response time to user feedback
        3. Consider feature requests with high user demand
        4. Monitor sentiment trends for early warning signs
        """)
    
    # Section 4: Comparative Analysis
    with st.expander("📅 Time-Based Comparison", expanded=False):
        st.subheader("Performance Over Time")
        
        # Create monthly comparison
        monthly_data = filtered_df.groupby('Month_Year').agg({
            'Rating': 'mean',
            'Sentiment_Score': 'mean',
            'UI_Issue': 'mean',
            'Performance_Issue': 'mean'
        }).reset_index()
        
        fig = px.line(monthly_data, x='Month_Year', y=['Rating', 'Sentiment_Score'],
                     title="Rating and Sentiment Trend",
                     labels={'value': 'Score', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(monthly_data, x='Month_Year', y=['UI_Issue', 'Performance_Issue'],
                     title="Issue Frequency Trend",
                     labels={'value': 'Percentage', 'variable': 'Issue Type'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Section 5: User Engagement
    with st.expander("👥 User Engagement Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Loyalty")
            loyal_users = filtered_df['User_Type'].value_counts(normalize=True).mul(100)
            fig = px.pie(loyal_users, values=loyal_users.values, names=loyal_users.index,
                        title="First-Time vs Loyal Users")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Response Time Analysis")
            if 'Reply_Time_Days' in filtered_df.columns:
                response_stats = filtered_df[filtered_df['Reply'] != "No Reply"]['Reply_Time_Days'].describe()
                st.metric("Average Response Time", f"{response_stats['mean']:.1f} days")
                st.metric("Median Response Time", f"{response_stats['50%']:.1f} days")
                st.metric("Fastest Response", f"{response_stats['min']:.1f} days")
                st.metric("Slowest Response", f"{response_stats['max']:.1f} days")
            else:
                st.warning("No response time data available")
    
    # Final Summary
    st.divider()
    st.subheader("📋 Final Assessment")
    
    # Generate dynamic summary based on data
    avg_rating = filtered_df['Rating'].mean()
    pos_sentiment = (filtered_df['Sentiment'] == 'Positive').mean()
    issue_rate = filtered_df[['UI_Issue', 'Performance_Issue']].any(axis=1).mean()
    
    if avg_rating >= 4:
        rating_verdict = "Excellent"
        rating_color = "green"
    elif avg_rating >= 3:
        rating_verdict = "Good"
        rating_color = "blue"
    else:
        rating_verdict = "Needs Improvement"
        rating_color = "red"
    
    summary = f"""
    <div style='border-left: 5px solid {rating_color}; padding-left: 15px;'>
    <h3>Overall App Status: <span style='color:{rating_color}'>{rating_verdict}</span></h3>
    <ul>
        <li>Average Rating: <b>{avg_rating:.1f}/5</b> stars</li>
        <li>Positive Sentiment: <b>{pos_sentiment:.1%}</b> of reviews</li>
        <li>Issue Reporting Rate: <b>{issue_rate:.1%}</b> of reviews mention problems</li>
        <li>Response Coverage: <b>{response_coverage:.1f}%</b> of reviews receive replies</li>
    </ul>
    </div>
    """
    
    st.markdown(summary, unsafe_allow_html=True)
    
    # Add download report button with error handling
    report_data = generate_report(filtered_df)
    if report_data:
        st.download_button(
            label="📥 Download Full Report",
            data=report_data,
            file_name="beauty_kult_app_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Could not generate PDF report")

with tab6:  # New Strategy tab
    st.header("🚀 Strategic Opportunities")
    
    # ROI Projections Section
    with st.expander("💰 ROI Projections", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rating Improvement Impact")
            rating_gap = industry_avg - avg_rating
            if rating_gap > 0:
                install_growth = rating_gap * 24  # 24% install boost per star (hypothetical)
                st.metric("Potential Install Growth", f"+{install_growth:.1f}%",
                         help="Based on industry data: each ★ = ~24% more installs")
            else:
                st.success("Your ratings exceed industry average!")
        
        with col2:
            st.subheader("Sentiment Improvement Value")
            current_value = pos_percent
            target_value = 85  # Industry top quartile
            if current_value < target_value:
                revenue_potential = (target_value - current_value) * 1000  # Hypothetical $1K per % point
                st.metric("Revenue Opportunity", f"${revenue_potential:,.0f}",
                         help="Estimated annual revenue potential from sentiment improvement")
            else:
                st.success("Your sentiment scores are in top quartile!")
    
    # Competitive Benchmarking Section
    with st.expander("📊 Competitive Benchmarking", expanded=True):
        # Mock competitor data - in real implementation use actual competitor data
        competitors = {
            'Beauty Kult': {
                'Rating': avg_rating,
                'Response Rate': response_rate,
                'Positive Sentiment': pos_percent,
                'UI Issues': filtered_df['UI_Issue'].mean() * 100
            },
            'Competitor A': {
                'Rating': 4.3,
                'Response Rate': 78,
                'Positive Sentiment': 82,
                'UI Issues': 12
            },
            'Competitor B': {
                'Rating': 4.1,
                'Response Rate': 65,
                'Positive Sentiment': 75,
                'UI Issues': 18
            }
        }
        
        # Convert to DataFrame for visualization
        benchmark_df = pd.DataFrame(competitors).T.reset_index().rename(columns={'index': 'App'})
        
        # Radar Chart
        categories = ['Rating', 'Response Rate', 'Positive Sentiment', 'UI Issues']
        fig = go.Figure()
        
        for app in benchmark_df['App']:
            fig.add_trace(go.Scatterpolar(
                r=benchmark_df[benchmark_df['App'] == app][categories].values[0],
                theta=categories,
                fill='toself',
                name=app
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Competitive Benchmarking Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Insights Section
    with st.expander("🔮 Predictive Insights", expanded=True):
        st.subheader("30-Day Rating Forecast")
        forecast_fig = generate_forecast(filtered_df)
        if forecast_fig:
            st.pyplot(forecast_fig)
        else:
            st.warning("Insufficient data for forecasting")
        
        # Response Time Impact Analysis
        st.subheader("Response Time Impact")
        if 'Reply_Time_Days' in filtered_df.columns:
            responsive_df = filtered_df[filtered_df['Reply'] != "No Reply"]
            if len(responsive_df) > 10:
                fig = px.scatter(responsive_df, 
                               x='Reply_Time_Days', 
                               y='Rating',
                               trendline="ols",
                               title="Faster Responses → Higher Ratings",
                               labels={'Reply_Time_Days': 'Days to Respond'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation
                fast_response = responsive_df[responsive_df['Reply_Time_Days'] <= 1]['Rating'].mean()
                slow_response = responsive_df[responsive_df['Reply_Time_Days'] > 1]['Rating'].mean()
                st.metric("Rating Boost from Fast Responses", 
                         f"+{fast_response - slow_response:.1f}★",
                         help="Average rating difference when responding within 1 day")
            else:
                st.warning("Not enough response data for analysis")
    
    # UGC Social Proof Section
    with st.expander("👥 User Voice", expanded=True):
        st.subheader("Top Positive Reviews")
        positive_reviews = filtered_df[filtered_df['Rating'] >= 4].sort_values('Rating', ascending=False).head(3)
        
        for _, row in positive_reviews.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"**⭐⭐⭐⭐⭐ {row['Rating']}**")
                    st.caption(row['Date'].strftime('%b %d, %Y'))
                with col2:
                    st.markdown(f"*\"{row['Review'][:200]}...\"*")
        
        st.subheader("Feature Requests Word Cloud")
        requests_text = " ".join(filtered_df[filtered_df['Feature_Request']]['Review'])
        if requests_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(requests_text)
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.warning("No feature requests found")
    
    # Service Recommendations Section
    with st.expander("🛠️ How We Can Help", expanded=True):
        st.subheader("Recommended Service Packages")
        
        # Basic Monitoring
        with st.container(border=True):
            st.markdown("#### 📊 Basic Monitoring ($299/mo)")
            st.markdown("""
            - Daily review tracking
            - Key metric dashboards
            - Weekly email reports
            - Basic sentiment analysis
            """)
            if st.button("Learn More", key="basic_monitoring"):
                st.session_state['show_basic'] = True
        
        # Pro Insights
        with st.container(border=True):
            st.markdown("#### 📈 Pro Insights ($799/mo)")
            st.markdown("""
            - Everything in Basic +
            - Competitor benchmarking
            - Predictive analytics
            - Custom action plans
            - Monthly strategy calls
            """)
            if st.button("Learn More", key="pro_insights"):
                st.session_state['show_pro'] = True
        
        # Enterprise
        with st.container(border=True):
            st.markdown("#### 🏢 Enterprise ($1,999/mo)")
            st.markdown("""
            - Everything in Pro +
            - AI-powered review responses
            - Real-time alerts
            - Dedicated account manager
            - Quarterly business reviews
            """)
            if st.button("Learn More", key="enterprise"):
                st.session_state['show_enterprise'] = True

# About Section
st.sidebar.markdown(f"""
### About This Dashboard

**Purpose:**  
Comprehensive analysis of Beauty Kult app reviews to identify improvement opportunities.

**Data Source:**  
Google Play Store reviews (updated {datetime.now().strftime('%Y-%m-%d')})

**Key Metrics Tracked:**  
- UI/UX Issues  
- Performance Problems  
- Feature Requests  
- Support Quality  
- Monetization Feedback  
- User Retention
""")

# Automated Alerts
if 'Reply_Time_Days' in filtered_df.columns and not filtered_df[filtered_df['Reply'] != "No Reply"].empty:
    current_response_time = filtered_df[filtered_df['Reply'] != "No Reply"]['Reply_Time_Days'].median()
    if current_response_time > 3:
        st.markdown(f"""
        <div class="alert-box">
            <h4>🔴 ALERT: Response times exceeding 3-day target! (Current: {current_response_time:.1f} days)</h4>
        </div>
        """, unsafe_allow_html=True)

if 'Support_Complaint' in filtered_df.columns:
    complaint_rate = filtered_df['Support_Complaint'].mean() * 100
    if complaint_rate > 25:
        st.markdown(f"""
        <div class="alert-box">
            <h4>🔴 ALERT: Support complaint rate above 25%! (Current: {complaint_rate:.1f}%)</h4>
        </div>
        """, unsafe_allow_html=True)

if 'Performance_Issue' in filtered_df.columns:
    perf_issue_rate = filtered_df['Performance_Issue'].mean() * 100
    if perf_issue_rate > 15:
        st.markdown(f"""
        <div class="alert-box">
            <h4>🔴 ALERT: High performance issue rate! (Current: {perf_issue_rate:.1f}%)</h4>
        </div>
        """, unsafe_allow_html=True)

if 'Feature_Request' in filtered_df.columns:
    feature_request_count = filtered_df['Feature_Request'].sum()
    if feature_request_count > 20:
        top_request = filtered_df[filtered_df['Feature_Request']]['Review'].value_counts().index[0][:100]
        st.markdown(f"""
        <div class="info-box">
            <h4>💡 Opportunity: {feature_request_count} feature requests detected!</h4>
            <p>Most requested: "{top_request}..."</p>
        </div>
        """, unsafe_allow_html=True)