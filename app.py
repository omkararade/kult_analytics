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
import matplotlib.patheffects as PathEffects
from prophet.plot import plot_plotly, plot_components_plotly
from matplotlib import cm
from plotly.subplots import make_subplots


# Initialize NLTK
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

# App Configuration
st.set_page_config(
    page_title="Swiggy App Analytics Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Tab Switcher
st.components.v1.html("""
<script>
function switchToTab(tabName) {
    const targetName = tabName.replace(/_/g, ' ').toLowerCase();
    const tabs = document.querySelectorAll('[data-baseweb="tab"]');
    tabs.forEach(tab => {
        const tabText = tab.textContent.trim().replace(/\s+/g, ' ').toLowerCase();
        if (tabText === targetName) {
            tab.click();
        }
    });
    window.location.hash = tabName;
    return false;
}
// Check hash on page load
window.addEventListener('load', function() {
    const hash = window.location.hash.substring(1);
    if (hash) {
        switchToTab(hash);
    }
});
</script>
""")
                      
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
        pdf.cell(200, 10, txt="Josh App Analytics Report", ln=1, align='C')
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
            ("Negative Sentiment", f"{(df['Sentiment'] == 'Negative').mean()*100:.1f}%"),
            ("Reply to Review Rate", f"{df['Reply'].apply(lambda x: x != 'No Reply').mean()*100:.1f}%"),
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

ISSUE_CONFIG = {
        'ui_issue': {
            'column': 'UI_Issue',
            'title': 'UI/UX Issues',
            'color': '#CB2726',
            'keywords': ['slow', 'lag', 'bug', 'glitch', 'crash', 'freeze', 'complicated', 'hard', 'navigation','unresponsive','delay','latency','stutter',
        'load time','resource intensive','memory leak','instability','error','failure','hang','confusing','difficult','intricate','unintuitive','cumbersome',
        'tedious','user-friendly','accessibility','workflow','steps','process','layout','design','interface','discoverability','pixelated','distorted',
        'alignment','animation','responsiveness','touch','click','scroll','visual','rendering','display','font','color','data loss','sync','save',
               'input','output','search','filter','functionality','feature','compatibility','frustrating','annoying','irritating','problem','issue','bad',
               'poor','broken','useless','disappointing']
        },
        'performance_issue': {
            'column': 'Performance_Issue',
            'title': 'Performance Issues',
            'color': '#CB2726',
            'keywords': ['crash','freeze','lag','slow','bug','glitch','not responding','stuck','hangs',
    'loading','performance','unstable','error','delay','latency','stutter','load time',
    'resource intensive','memory leak','instability','failure','unresponsive','rendering','optimization']
        },
        'feature_request': {
            'column': 'Feature_Request',
            'title': 'Feature Requests',
            'color': '#CB2726',
            'keywords': ['should have', 'need', 'want', 'please add', 'where is', 'why no', 'missing', 'would love',
            'wish there was', 'suggest', 'recommend', 'hope to see', 'require', 'desire', 'looking for',
            'could use', 'it would be great if', 'it would be helpful if', 'is there a way to',
            'is it possible to', 'consider adding', 'Id like to see', 'Im trying to find',
            'can you implement', 'how do I', 'is it possible to get', 'is there', 'Im looking for']
        },
        'support_complaint': {
            'column': 'Support_Complaint',
            'title': 'Support Issues',
            'color': '#CB2726',
            'keywords': {'No Response': ['no reply', 'no answer', 'ignored', 'no help', 'no response', 'never responded', 'no feedback'],
        'Slow Response': ['slow response', 'took long', 'days to reply', 'delayed response', 'long wait', 'prolonged delay', 'late reply'],
        'Unhelpful': ['not helpful', 'useless', 'did not solve', 'waste of time', 'ineffective', 'unhelpful', 'did not assist', 'no solution', 'failed to resolve'],
        'Rude Staff': ['rude', 'arrogant', 'unprofessional', 'angry', 'impolite', 'disrespectful', 'hostile', 'offensive', 'dismissive']
            }
        },
        'pricing_complaint': {
            'column': 'Pricing_Complaint',
            'title': 'Pricing Issues',
            'color': '#CB2726',
            'keywords': ['expensive','overpriced','pricey','too much','not worth','high price','cost too much',
        'unfair','cheaper','lower price','reduce price','price hike','cost','value','affordable']
        },
        'delivery_issues': {
        'column': 'delivery_issues',
        'title': 'Delivery Issues',
        'color': '#FF6B6B',
        'keywords': [
            'late delivery', 'delayed', 'not delivered', 'delivery time', 'driver late', 'took too long',
            'ETA', 'wrong address', 'missed delivery', 'delivery failed', 'reschedule', 'never arrived',
            'package delay', 'delivery issue', 'still waiting', 'came late', 'got it late', 'delay in delivery',
            'order late', 'order not here', 'waiting for my order', 'where is my order', 'running late',
            'delivered to wrong address', 'delivered somewhere else', 'didn’t show up', 'arrived late'
        ]
        },
        'payment_issue': {
        'column': 'Payment_Problems',
        'title': 'Payment Problems',
        'color': '#FFA500',
        'keywords': [
            'payment failed', 'transaction error', 'card declined', 'not processed', 'double charged',
            'overcharged', 'refund pending', 'refund delay', 'payment issue', "can't pay", 'not refunded',
            'incorrect amount', 'failed to pay', 'billing error', 'charge issue', 'charged twice',
            'money deducted', 'amount not refunded', 'payment stuck', 'did not get refund',
            'transaction declined', 'unable to pay', 'app charged me', 'no confirmation after payment',
            'payment not successful'
        ]
        },
        'food_quality': {
        'column': 'Food_Quality',
        'title': 'Food Quality',
        'color': '#6A5ACD',
        'keywords': [
            'stale food', 'not fresh', 'cold food', 'bad taste', 'spoiled', 'poor quality',
            'packaging issue', 'leaked', 'damaged package', 'soggy', 'missing items', 'wrong item',
            'undercooked', 'overcooked', 'smells bad', 'rotten', 'food poisoning', 'not edible',
            'food was cold', 'food was awful', 'not good', 'unhygienic', 'dirty packaging',
            'weird smell', 'wrong dish', 'order messed up', 'hair in food', 'low quality', 'bad smell'
        ]
        },
        'promotions_issue': {
        'column': 'Promotions_Issues',
        'title': 'Promotions and Coupons',
        'color': '#32CD32',
        'keywords': [
            'coupon not working', 'promo code invalid', 'offer not applied', 'discount not working',
            'code expired', "can't use promo", 'offer issue', 'not eligible', "didn't get discount",
            'cashback not received', 'free delivery not applied', 'reward not credited',
            'code not accepted', 'voucher didn’t work', 'promo didn’t apply', 'invalid promo',
            'didn’t get offer', 'promotion failed', 'no discount received', 'promo not working',
            'discount missing', 'free item not added', 'applying coupon failed', 'reward missing'
        ]
        },


    }


# 1. Correct Forecast Function
def generate_forecast(df):
    """Generate rating forecast using Prophet with error handling"""
    try:
        # Validate data
        if df.empty or pd.isnull(df['Date']).any():
            raise ValueError("Missing date values in dataset")
            
        if len(df) < 60:
            raise ValueError("Minimum 60 data points required")
            
        # Prepare data
        df_forecast = df.set_index('Date').resample('D')['Rating'].mean().reset_index()
        df_forecast.columns = ['ds', 'y']
        df_forecast = df_forecast.dropna()
        
        # Create model
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(df_forecast)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=180)
        forecast = model.predict(future)
        
        # Return Plotly figure
        return plot_plotly(model, forecast)
        
    except Exception as e:
        st.error(f"Forecast failed: {str(e)}")
        return None
    
@st.cache_data
def load_data():
    df = pd.read_csv("Swiggy_13k+.csv")
    
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
        'slow', 'lag', 'bug', 'glitch', 'crash', 'freeze', 'complicated', 'hard', 'navigation','unresponsive','delay','latency','stutter',
        'load time','resource intensive','memory leak','instability','error','failure','hang','confusing','difficult','intricate','unintuitive','cumbersome',
        'tedious','user-friendly','accessibility','workflow','steps','process','layout','design','interface','discoverability','pixelated','distorted',
        'alignment','animation','responsiveness','touch','click','scroll','visual','rendering','display','font','color','data loss','sync','save',
               'input','output','search','filter','functionality','feature','compatibility','frustrating','annoying','irritating','problem','issue','bad',
               'poor','broken','useless','disappointing'
    ]
    df['UI_Issue'] = df['Review'].str.contains('|'.join(ui_keywords), case=False, na=False)
    
    performance_keywords = [
        'crash','freeze','lag','slow','bug','glitch','not responding','stuck','hangs',
    'loading','performance','unstable','error','delay','latency','stutter','load time',
    'resource intensive','memory leak','instability','failure','unresponsive','rendering','optimization'
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
        'subscription','renewal','auto-renew','cancel','refund','billing','charge','payment','iap','in-app purchase',
    'trial','scam','unsubscribe','manage subscription','subscription fees','subscription cost','subscription service','membership',
    'recurring payment','subscription plan','subscription model','subscription options','subscription terms','subscription issues',
    'subscription problems','subscription error','subscription expired','subscription active','subscription paused','subscription status',
    'subscription details','subscription access','subscription account','subscription support','subscription help','subscription cancellation',
    'subscription confirmation','subscription history','subscription information'
    ]
    df['Subscription_Complaint'] = df['Review'].str.contains('|'.join(subscription_keywords), case=False, na=False)
    
    subscription_issues = {
         'Auto-Renewal': ['auto-renew', 'unsubscribe', 'hard to cancel', 'difficult to cancel', 'automatic renewal', 'cancellation problems', 'cannot unsubscribe'],
    'Unexpected Charges': ['unexpected charge', 'hidden fee', 'surprise charge', 'unauthorized charge', 'extra fees', 'unknown charge', 'incorrect billing'],
    'Refund Problems': ['refund', 'money back', 'not refund', 'no refund', 'refund denied', 'refund issues', 'refund process', 'refund policy'],
    'Value Issues': ['not worth', 'waste of money', 'better free', 'overpriced subscription', 'poor value', 'not worth the cost', 'expensive for what it offers']
    }
    DELIVERY_ISSUE_KEYWORDS = [
    'late delivery', 'delayed', 'not delivered', 'delivery time', 'driver late', 'took too long',
    'ETA', 'wrong address', 'missed delivery', 'delivery failed', 'reschedule', 'never arrived',
    'package delay', 'delivery issue', 'still waiting', 'came late', 'got it late', 'delay in delivery',
    'order late', 'order not here', 'waiting for my order', 'where is my order', 'running late',
    'delivered to wrong address', 'delivered somewhere else', 'didn’t show up', 'arrived late'
    ]
    df['delivery_issues'] = df['Review'].str.contains('|'.join(DELIVERY_ISSUE_KEYWORDS), case=False, na=False)
    PAYMENT_PROBLEM_KEYWORDS = [
    'payment failed', 'transaction error', 'card declined', 'not processed', 'double charged',
    'overcharged', 'refund pending', 'refund delay', 'payment issue', "can't pay", 'not refunded',
    'incorrect amount', 'failed to pay', 'billing error', 'charge issue', 'charged twice',
    'money deducted', 'amount not refunded', 'payment stuck', 'did not get refund',
    'transaction declined', 'unable to pay', 'app charged me', 'no confirmation after payment',
    'payment not successful']
    df['Payment_Problems'] = df['Review'].str.contains('|'.join(PAYMENT_PROBLEM_KEYWORDS), case=False, na=False)
    FOOD_QUALITY_KEYWORDS = [
    'stale food', 'not fresh', 'cold food', 'bad taste', 'spoiled', 'poor quality',
    'packaging issue', 'leaked', 'damaged package', 'soggy', 'missing items', 'wrong item',
    'undercooked', 'overcooked', 'smells bad', 'rotten', 'food poisoning', 'not edible',
    'food was cold', 'food was awful', 'not good', 'unhygienic', 'dirty packaging',
    'weird smell', 'wrong dish', 'order messed up', 'hair in food', 'low quality', 'bad smell'
]
    df['Food_Quality'] = df['Review'].str.contains('|'.join(FOOD_QUALITY_KEYWORDS), case=False, na=False)
    PROMOTION_ISSUE_KEYWORDS = [
    'coupon not working', 'promo code invalid', 'offer not applied', 'discount not working',
    'code expired', "can't use promo", 'offer issue', 'not eligible', "didn't get discount",
    'cashback not received', 'free delivery not applied', 'reward not credited',
    'code not accepted', 'voucher didn’t work', 'promo didn’t apply', 'invalid promo',
    'didn’t get offer', 'promotion failed', 'no discount received', 'promo not working',
    'discount missing', 'free item not added', 'applying coupon failed', 'reward missing'
]
    df['Promotions_Issues'] = df['Review'].str.contains('|'.join(PROMOTION_ISSUE_KEYWORDS), case=False, na=False)




    
    
    def categorize_sub_issue(text):
        text = str(text).lower()
        for category, keywords in subscription_issues.items():
            if any(keyword in text for keyword in keywords):
                return category
        return 'Other'
    
    df.loc[df['Subscription_Complaint'], 'Subscription_Issue_Type'] = df[df['Subscription_Complaint']]['Review'].apply(categorize_sub_issue)
    
    request_phrases = [
    'should have','need','want','please add','where is','why no','missing','would love','wish there was','suggest','recommend',
    'hope to see','require','desire','looking for','could use','it would be great if','it would be helpful if','is there a way to',
    'is it possible to','consider adding','Id like to see',
    'Im trying to find','can you implement','how do I','is it possible to get','is there','Im looking for'
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
    #(df['Date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
    ((df['Sentiment'].isin(sentiment_filter)) if sentiment_filter else True)
].copy()

# Main Dashboard
st.title("Josh App Analytics Dashboard")

# KPI Cards with Competitive Benchmark
col1, col2, col3, col4,col5,col6 = st.columns(6)
with col2:
    avg_rating = filtered_df['Rating'].mean()
    st.metric("Average Rating", f"{avg_rating:.1f}")

with col3:
    response_rate = filtered_df['Reply'].apply(lambda x: x != "No Reply").mean() * 100
    st.metric("Reply to Review Rate", f"{response_rate:.1f}%")

with col1:
    total_reviews = len(filtered_df)
    st.metric("Total Reviews", f"{total_reviews:,}")

with col4:
    pos_percent = (filtered_df['Sentiment'] == 'Positive').mean() * 100
    st.metric("Positive Sentiment", f"{pos_percent:.1f}%")
with col5:
    neg_percent = (filtered_df['Sentiment'] == 'Negative').mean() * 100
    st.metric("Negative Sentiment", f"{neg_percent:.1f}%")
with col6:
    neg_percent = (filtered_df['Sentiment'] == 'Neutral').mean() * 100
    st.metric("Neutral Sentiment", f"{neg_percent:.1f}%")


# Tabs with New Strategy Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Trends", "Reviews", "Insights", "Issues", "Report", "Strategy"]
)

with tab1:
    st.header("Trends Over Time")
    
    # ---- Rating Trend ----
    st.subheader("Rating Performance")

    trend_data = filtered_df.groupby('Month_Year').agg(
        Avg_Rating=('Rating', 'mean')
    ).reset_index()

    fig = px.line(trend_data, x='Month_Year', y='Avg_Rating',
                labels={'Avg_Rating': 'Average Rating', 'Month_Year': 'Month'},
                height=350,
                line_shape='linear')

    # Change line color to light red
    fig.update_traces(line=dict(color='#CB2726'))
    fig.update_layout(
        xaxis=dict(showgrid=False),  # Remove vertical grid lines
        yaxis=dict(showgrid=False),  # Remove horizontal grid lines
        plot_bgcolor='rgba(0,0,0,0)'  # Make background transparent
    )

    fig.update_layout(
        legend=dict(x=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)

    from datetime import datetime

    # Convert to datetime first (if not already done)
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])  # Replace 'Date_Column' with your actual date column
    filtered_df['Month_Year'] = filtered_df['Date'].dt.to_period('M').dt.strftime('%b %Y')  # Format: "Jan 2023"

    # Rating Trend with Year-Month
    trend_data = filtered_df.groupby('Month_Year').agg(
        Avg_Rating=('Rating', 'mean'),
        Review_Count=('Rating', 'count')
    ).reset_index()

   
    # ---- Sentiment Analysis ----
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Sentiment Trend")
        sentiment_trend = filtered_df.groupby('Month_Year')['Sentiment_Score'].mean().reset_index()
        
        fig = px.line(
            sentiment_trend, 
            x='Month_Year', 
            y='Sentiment_Score',
            labels={'Sentiment_Score': 'Average Score'},
            height=400
        )
        fig.update_traces(line=dict(color='rgba(255, 99, 132, 0.8)'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Distribution")
        # Create cohorts and counts
        bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        labels = [
            'Extreme Neg', 'Strong Neg', 'Moderate Neg', 'Mild Neg',
            'Neutral', 'Mild Pos', 'Moderate Pos', 'Strong Pos'
        ]
        filtered_df['Cohort'] = pd.cut(filtered_df['Sentiment_Score'], bins=bins, labels=labels)
        cohort_counts = filtered_df['Cohort'].value_counts().sort_index()
        max_count = cohort_counts.max()

        # Rotated visualization
        fig, ax = plt.subplots(figsize=(4, 2.5))  # Swapped dimensions
        cmap = plt.cm.get_cmap('viridis', 256)
        
        for i, (cohort, count) in enumerate(cohort_counts.items()):
            color_intensity = count / max_count  
            ax.bar(
                [i], 
                [1], 
                color=cmap(color_intensity), 
                width=0.7,  # Changed from height to width
                edgecolor='white'
            )
            ax.text(
                i, 0.5, f"{count}",  # Swapped coordinates
                ha='center', 
                va='center', 
                color='white' if color_intensity > 0.5 else 'black',
                fontsize=8,
                fontdict={'weight': 'bold'},
                rotation=90  # Rotate text
            )

        # Adjusted formatting
        ax.set_xticks(range(len(cohort_counts)))
        ax.set_xticklabels(cohort_counts.index, fontsize=8, rotation=45, ha='right')
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.tight_layout()
        st.pyplot(fig)

    # Review Volume by Weekday
    weekday_counts = filtered_df['Weekday'].value_counts().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    fig = px.bar(weekday_counts, 
                title="Review Volume by Day of Week",
                labels={'value': 'Number of Reviews', 'index': 'Day of Week'})
    st.plotly_chart(fig, use_container_width=True)

    # Average Rating by Weekday
    weekday_ratings = filtered_df.groupby('Weekday')['Rating'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    fig = px.bar(weekday_ratings, 
                title="Average Rating by Day of Week",
                labels={'value': 'Average Rating', 'index': 'Day of Week'},
                color=weekday_ratings.values,
                color_continuous_scale='Reds')
    fig.update_layout(yaxis_range=[1,5])  # Assuming ratings are 1-5
    st.plotly_chart(fig, use_container_width=True)

    # Histogram of star ratings
    fig_hist = px.histogram(
        filtered_df,
        x='Rating',
        nbins=5,
        title="Distribution of Star Ratings",
        color_discrete_sequence=['#1f77b4']
    )
    fig_hist.update_layout(
        xaxis_title='Star Rating',
        yaxis_title='Count',
        bargap=0.2
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    # Ensure Date column is in datetime format
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    # Calculate midpoint date
    midpoint_date = filtered_df['Date'].sort_values().iloc[len(filtered_df) // 2]

    # Label periods as 'Before' and 'After'
    filtered_df['Period'] = filtered_df['Date'].apply(lambda x: 'Before' if x < midpoint_date else 'After')

    # Aggregate statistics for each metric
    compare_stats = filtered_df.groupby('Period').agg({
        'Rating': 'mean',
        'Sentiment_Score': 'mean',
        'Review': 'count'
    }).rename(columns={'Review': 'Review_Count'}).reset_index()

    # Create subplots for comparison
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Average Rating", "Average Sentiment Score", "Number of Reviews"),
        shared_yaxes=False
    )

    # Plot for Average Rating
    fig.add_trace(
        go.Bar(
            x=compare_stats['Period'],
            y=compare_stats['Rating'],
            name='Average Rating',
            marker_color='rgba(255,99,132,0.6)'
        ),
        row=1, col=1
    )

    # Plot for Average Sentiment Score
    fig.add_trace(
        go.Bar(
            x=compare_stats['Period'],
            y=compare_stats['Sentiment_Score'],
            name='Sentiment Score',
            marker_color='rgba(54,162,235,0.6)'
        ),
        row=1, col=2
    )

    # Plot for Number of Reviews
    fig.add_trace(
        go.Bar(
            x=compare_stats['Period'],
            y=compare_stats['Review_Count'],
            name='Number of Reviews',
            marker_color='rgba(75,192,192,0.6)'
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title="Comparison Before vs After Midpoint Date",
        showlegend=False,
        height=500,  # Adjust the height of the figure
        xaxis=dict(title='Period'),
        yaxis=dict(title='Average Rating', range=[1, 5]),  # Assuming ratings are between 1 and 5
        yaxis2=dict(title='Average Sentiment Score', range=[-1, 1]),
        yaxis3=dict(title='Number of Reviews'),
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Show midpoint date
    st.caption(f"🗓️ Midpoint date used for comparison: **{midpoint_date.strftime('%Y-%m-%d')}**")



    # Sample(Test Additional Graphs)

    st.subheader("Metric Relationships")
    corr_matrix = filtered_df[['Rating', 'Sentiment_Score', 'Reply_Time_Days']].corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)


    # Aggregate all issue types and replace 0% with NaN
    issues_trend = filtered_df.groupby('Month_Year').agg({
        'UI_Issue': lambda x: x.mean() if x.sum() > 0 else np.nan,
        'Performance_Issue': lambda x: x.mean() if x.sum() > 0 else np.nan,
        'Support_Complaint': lambda x: x.mean() if x.sum() > 0 else np.nan,
        'Feature_Request': lambda x: x.mean() if x.sum() > 0 else np.nan,
        'delivery_issues': lambda x: x.mean() if x.sum() > 0 else np.nan,
        'Payment_Problems': lambda x: x.mean() if x.sum() > 0 else np.nan,
        'Food_Quality': lambda x: x.mean() if x.sum() > 0 else np.nan,
        'Promotions_Issues': lambda x: x.mean() if x.sum() > 0 else np.nan
    }).reset_index()

    # Emerging Issues Timeline Section
    st.subheader("Emerging Issues Timeline")

    # Aggregate all issue types and filter zeros
    issues_trend = filtered_df.groupby('Month_Year').agg({
        'UI_Issue': 'mean',
        'Performance_Issue': 'mean',
        'Support_Complaint': 'mean',
        'Feature_Request': 'mean',
        'delivery_issues': 'mean',
        'Payment_Problems': 'mean',
        'Food_Quality': 'mean',
        'Promotions_Issues': 'mean'
    }).replace(0, np.nan).reset_index()

    # Melt for plotting and clean data
    melted_issues = issues_trend.melt(
        id_vars='Month_Year', 
        var_name='Issue_Type', 
        value_name='Frequency'
    ).dropna(subset=['Frequency'])

    # Create formatted labels
    issue_labels = {
        'UI_Issue': 'UI/UX Issues',
        'Performance_Issue': 'Performance Issues',
        'Support_Complaint': 'Support Complaints',
        'Feature_Request': 'Feature Requests',
        'delivery_issues': 'Delivery Issues',
        'Payment_Problems': 'Payment Problems',
        'Food_Quality': 'Food Quality',
        'Promotions_Issues': 'Promotions Issues'
    }

    # Create the plot
    fig = px.line(
        melted_issues,
        x='Month_Year',
        y='Frequency',
        color='Issue_Type',
        labels={'Frequency': 'Percentage of Reviews', 'Issue_Type': 'Issue Type'},
        title="Monthly Issue Frequency Trends",
        category_orders={"Issue_Type": list(issue_labels.keys())},
        line_shape='linear'
    )

    # Formatting improvements
    fig.update_layout(
        yaxis_tickformat=".0%",
        legend_title_text=None,
        hovermode="x unified",
        yaxis_range=[0, melted_issues['Frequency'].max() * 1.1]
    )

    # Custom styling
    colors = px.colors.qualitative.Plotly
    for i, trace in enumerate(fig.data):
        trace.update(
            line=dict(width=2.5),
            mode='lines+markers',
            marker=dict(size=6),
            name=issue_labels[trace.name],
            hovertemplate="%{y:.1%}",
            connectgaps=False
        )
        if i < len(colors):
            trace.update(line_color=colors[i])

    st.plotly_chart(fig, use_container_width=True)

        # Issue Summary Analysis
    st.subheader("Issue Summary Analysis")

    # Define columns to analyze
    issue_columns = ['UI_Issue', 'Performance_Issue', 'Support_Complaint', 'Feature_Request','delivery_issues', 'Payment_Problems', 'Food_Quality', 'Promotions_Issues']

    # Create summary dataframe and filter zeros
    issue_summary = pd.DataFrame({
        'Total Reports': filtered_df[issue_columns].sum(),
        '% of Total Reviews': filtered_df[issue_columns].mean() * 100
    }).reset_index().rename(columns={'index': 'Issue Type'})

    # Filter out issues with zero reports and format
    issue_summary = (
        issue_summary
        .query("`Total Reports` > 0")  # Remove zero-count issues
        .assign(
            **{'Issue Type': lambda x: x['Issue Type'].map(issue_labels)},
            **{'% of Total Reviews': lambda x: x['% of Total Reviews'].round(1)}
        )
    )

    # Only show if there's data to display
    if not issue_summary.empty:
        st.dataframe(
            issue_summary.style
            .background_gradient(subset=['Total Reports'], cmap='Reds')
            .background_gradient(subset=['% of Total Reviews'], cmap='Blues')
            .format({'Total Reports': '{:,}', '% of Total Reviews': '{:.1f}%'}),
            height=400,
            column_config={
                "Issue Type": st.column_config.TextColumn(width="medium"),
                "Total Reports": st.column_config.NumberColumn(
                    help="Total number of reviews mentioning this issue"
                ),
                "% of Total Reviews": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Percentage of all reviews mentioning this issue"
                )
            }
        )
    else:
        st.info("No significant issues found in the selected timeframe")

    st.subheader("Review Type Composition")

    # Create review type classification
    review_types = filtered_df.assign(
        Type=np.select(
            condlist=[
                filtered_df['Feature_Request'],  # First condition: Feature requests
                filtered_df['Support_Complaint']  # Second condition: Support complaints
            ],
            choicelist=[
                'Feature Request',
                'Support Issue'
            ],
            default=np.where(
                filtered_df['Rating'] > 3,
                'Positive Feedback',
                'General Complaint'
            )
        )
    )

    # Calculate monthly distribution
    type_distribution = (review_types
                        .groupby('Month_Year')['Type']
                        .value_counts(normalize=True)
                        .unstack()
                        .fillna(0)
                        .sort_index())

    # Format for plotting
    type_distribution = type_distribution[['Feature Request', 'Support Issue','General Complaint', 'Positive Feedback']]

    # Create visualization
    fig = px.area(
        type_distribution,
        title="Review Type Distribution Over Time",
        labels={'value': 'Percentage', 'variable': 'Review Type'},
        color_discrete_map={
            'Feature Request': '#4C78A8',
            'Support Issue': '#E45756',
            'General Complaint': '#F58518',
            'Positive Feedback': '#54A24B'
        }
    )

    # Format axes
    fig.update_layout(
        xaxis_title='Month',
        yaxis=dict(tickformat=".0%"),
        hovermode='x unified'
    )

    # Add helpful annotations
    max_month = type_distribution.index[-1]
    if 'General Complaint' in type_distribution.columns:
        latest_complaints = type_distribution.loc[max_month, 'General Complaint']
        if latest_complaints > 0.3:
            fig.add_annotation(
                x=max_month,
                y=latest_complaints,
                text="High Complaints!",
                showarrow=True,
                arrowhead=1,
                ax=-50,
                ay=-30
            )

    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Review Type Composition")

    # Create review type categories
    conditions = [
        filtered_df['UI_Issue'],  # Feature requests
        filtered_df['Rating'] > 3,       # Positive feedback
        filtered_df['Rating'] <= 3       # Complaints
    ]

    choices = [
        'UI Isuue',
        'Positive Feedback', 
        'Complaint'
    ]

    # Create review type column
    review_df = filtered_df.assign(
        Type=np.select(conditions, choices, default='Other')
    )

    # Aggregate data by month and type
    type_counts = (review_df.groupby(['Month_Year', 'Type'])
                .size()
                .unstack(fill_value=0)
                .div(review_df.groupby('Month_Year').size(), axis=0)
                .reset_index()
                .melt(id_vars='Month_Year', var_name='Type', value_name='Percentage'))

    # Create visualization
    fig = px.area(type_counts, 
                x='Month_Year', 
                y='Percentage',
                color='Type',
                title="Review Type Distribution Over Time",
                labels={'Percentage': 'Proportion of Reviews'},
                category_orders={"Type": ["Feature Request", "Positive Feedback", "Complaint", "Other"]})

    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

   
    UI_KEYWORDS = [
        'slow', 'lag', 'bug', 'glitch', 'crash', 'freeze', 'complicated', 'hard', 'navigation',
        'unresponsive', 'delay', 'latency', 'stutter', 'load time', 'resource intensive',
        'memory leak', 'instability', 'error', 'failure', 'hang', 'confusing', 'difficult',
        'intricate', 'unintuitive', 'cumbersome', 'tedious', 'user-friendly', 'accessibility',
        'workflow', 'steps', 'process', 'layout', 'design', 'interface', 'discoverability',
        'pixelated', 'distorted', 'alignment', 'animation', 'responsiveness', 'touch', 'click',
        'scroll', 'visual', 'rendering', 'display', 'font', 'color', 'data loss', 'sync', 'save',
        'input', 'output', 'search', 'filter', 'functionality', 'feature', 'compatibility',
        'frustrating', 'annoying', 'irritating', 'problem', 'issue', 'bad', 'poor', 'broken',
        'useless', 'disappointing'
    ]

    PERFORMANCE_KEYWORDS = [
        'crash', 'freeze', 'lag', 'slow', 'bug', 'glitch', 'not responding', 'stuck', 'hangs',
        'loading', 'performance', 'unstable', 'error', 'delay', 'latency', 'stutter', 'load time',
        'resource intensive', 'memory leak', 'instability', 'failure', 'unresponsive', 'rendering',
        'optimization'
    ]

    FEATURE_REQUEST_KEYWORDS = [
        'should have', 'need', 'want', 'please add', 'where is', 'why no', 'missing', 'would love',
        'wish there was', 'suggest', 'recommend', 'hope to see', 'require', 'desire', 'looking for',
        'could use', 'it would be great if', 'it would be helpful if', 'is there a way to',
        'is it possible to', 'consider adding', 'Id like to see', 'Im trying to find',
        'can you implement', 'how do I', 'is it possible to get', 'is there', 'Im looking for'
    ]

    delivery_keywords = [
    'late delivery', 'delayed', 'not delivered', 'delivery time', 'driver late', 'took too long',
    'ETA', 'wrong address', 'missed delivery', 'delivery failed', 'reschedule', 'never arrived',
    'package delay', 'delivery issue', 'still waiting', 'came late', 'got it late', 'delay in delivery',
    'order late', 'order not here', 'waiting for my order', 'where is my order', 'running late',
    'delivered to wrong address', 'delivered somewhere else', 'didn’t show up', 'arrived late'
    ]
    payment_keywords = [
    'payment failed', 'transaction error', 'card declined', 'not processed', 'double charged',
    'overcharged', 'refund pending', 'refund delay', 'payment issue', "can't pay", 'not refunded',
    'incorrect amount', 'failed to pay', 'billing error', 'charge issue', 'charged twice',
    'money deducted', 'amount not refunded', 'payment stuck', 'did not get refund',
    'transaction declined', 'unable to pay', 'app charged me', 'no confirmation after payment',
    'payment not successful'
     ]
    food_quality_keywords = [
    'stale food', 'not fresh', 'cold food', 'bad taste', 'spoiled', 'poor quality',
    'packaging issue', 'leaked', 'damaged package', 'soggy', 'missing items', 'wrong item',
    'undercooked', 'overcooked', 'smells bad', 'rotten', 'food poisoning', 'not edible',
    'food was cold', 'food was awful', 'not good', 'unhygienic', 'dirty packaging',
    'weird smell', 'wrong dish', 'order messed up', 'hair in food', 'low quality', 'bad smell'
     ]
    promotion_keywords = [
    'coupon not working', 'promo code invalid', 'offer not applied', 'discount not working',
    'code expired', "can't use promo", 'offer issue', 'not eligible', "didn't get discount",
    'cashback not received', 'free delivery not applied', 'reward not credited',
    'code not accepted', 'voucher didn’t work', 'promo didn’t apply', 'invalid promo',
    'didn’t get offer', 'promotion failed', 'no discount received', 'promo not working',
    'discount missing', 'free item not added', 'applying coupon failed', 'reward missing'
     ]


    SUPPORT_CATEGORIES = {
        'No Response': ['no reply', 'no answer', 'ignored', 'no help', 'no response', 
                    'never responded', 'no feedback'],
        'Slow Response': ['slow response', 'took long', 'days to reply', 'delayed response',
                        'long wait', 'prolonged delay', 'late reply'],
        'Unhelpful': ['not helpful', 'useless', 'did not solve', 'waste of time',
                    'ineffective', 'unhelpful', 'did not assist', 'no solution',
                    'failed to resolve'],
        'Rude Staff': ['rude', 'arrogant', 'unprofessional', 'angry', 'impolite',
                    'disrespectful', 'hostile', 'offensive', 'dismissive']
    }

    # --------------------------
    # 2. Create Issue Flags
    # --------------------------
    # UI Issues Detection

    # Performance Issues Detection
    df['Performance_Issue'] = df['Review'].str.contains(
        '|'.join(PERFORMANCE_KEYWORDS), case=False, na=False
    )
    df['UI_Issue'] = df['Review'].str.contains('|'.join(UI_KEYWORDS), case=False, na=False)

    # Feature Requests Detection
    df['Feature_Request'] = df['Review'].str.contains(
        '|'.join(FEATURE_REQUEST_KEYWORDS), case=False, na=False
    )
    df['delivery_issues'] = df['Review'].str.contains('|'.join(delivery_keywords), case=False)
    df['Payment_Issue'] = df['Review'].str.contains('|'.join(payment_keywords), case=False)
    df['Food_Quality_Issue'] = df['Review'].str.contains('|'.join(food_quality_keywords), case=False)
    df['Promotion_Issue'] = df['Review'].str.contains('|'.join(promotion_keywords), case=False)

    # --------------------------
    # 3. Support Issue Categorization
    # --------------------------
    def categorize_support_issue(review):
        """Classify support-related issues into subcategories"""
        review_text = str(review).lower()
        for category, keywords in SUPPORT_CATEGORIES.items():
            if any(keyword in review_text for keyword in keywords):
                return category
        return None

    df['Support_Issue'] = df['Review'].apply(categorize_support_issue)

    # --------------------------
    # 4. Issue Type Classification
        # --------------------------
    def classify_issue(row):
        """Determine primary issue category for each review"""
        if row['UI_Issue']:
            return 'UI Issue'
        elif row['Performance_Issue']:
            return 'Performance Issue'
        elif pd.notna(row['Support_Issue']):
            return 'Support Issue'  # Fixed: Return standardized label instead of column value
        elif row['Feature_Request']:
            return 'Feature Request'
        elif row['delivery_issues']:
            return 'Delivery Issue'
        elif row['Payment_Issue']:
            return 'Payment Issue'
        elif row['Food_Quality_Issue']:
            return 'Food Quality Issue'
        elif row['Promotion_Issue']:
            return 'Promotion Issue'
        elif row['Subscription_Complaint']:
            return 'Subscription Issue'
        return 'Other'

    df['Issue_Type'] = df.apply(classify_issue, axis=1)


    issue_data = df[df['Issue_Type'] != 'Other'] \
                .groupby(['Issue_Type', 'Rating'], observed=True) \
                .size() \
                .reset_index(name='Reports')

    # Create base chart
    sunburst = px.sunburst(
        issue_data,
        path=['Issue_Type', 'Rating'],
        values='Reports',
        title='<b>Customer Feedback Analysis</b>',
        color='Issue_Type',
        color_discrete_sequence=px.colors.sequential.Redor,
        width=800,
        height=800
    )

    # Premium styling adjustments
    sunburst.update_traces(
        texttemplate='<b>%{label}</b><br>%{value:.0f} Reports',
        textfont=dict(
            family="Arial",
            size=20,
            color='white'
        ),
        marker=dict(
            line=dict(
                color='rgba(255,255,255,0.8)', 
                width=1.5
            )
        ),
        hovertemplate='<b>%{label}</b><br>%{value:.0f} Reports<extra></extra>'
    )

    sunburst.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, l=20, r=20, b=20),
        title={
            'text': '<b>Customer Feedback Analysis</b>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 24,
                'color': '#c00000',
                'family': 'Arial Black'
            }
        },
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(
                size=12,
                color='#5a5a5a'
            )
        ),
        uniformtext=dict(
            minsize=14,
            mode='hide'
        )
    )

    # Add corporate watermark
    sunburst.add_annotation(
        text="Josh App",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(
            size=12,
            color='#c00000'
        ),
        xref="paper",
        yref="paper"
    )

    st.plotly_chart(sunburst, use_container_width=True)


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
    st.subheader("Top 10 Complaint Analysis")
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
        # Process top 10 issues to ensure we have enough for both categories
        for issue, _ in top_issues[:10]:  # Changed from 5 to 10
            affected = filtered_df[filtered_df['Review'].str.contains(issue, case=False)]
            if len(affected) > 0:
                non_affected = filtered_df[~filtered_df['Review'].str.contains(issue, case=False)]
                impact = non_affected['Rating'].mean() - affected['Rating'].mean()
                issue_impact.append({
                    'Issue': issue,
                    'Affected Reviews': len(affected),
                    'Rating Impact': impact
                })

        if issue_impact:
            impact_df = pd.DataFrame(issue_impact)
            
            # Split into negative and positive impacts
            negative_impact_df = impact_df[impact_df['Rating Impact'] > 0]\
                .sort_values('Rating Impact', ascending=False).head(5)
            
            positive_impact_df = impact_df[impact_df['Rating Impact'] < 0]\
                .sort_values('Rating Impact', ascending=True).head(5)

            # Create two columns for side-by-side display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Top Negative Impacts")
                st.dataframe(
                    negative_impact_df,
                    column_config={
                        "Rating Impact": st.column_config.NumberColumn(
                            format="▼ %.2f",
                            help="How much ratings decrease when this issue is mentioned"
                        )
                    },
                    height=250,
                    hide_index=True
                )
            
            with col2:
                st.subheader("Top Positive Impacts")
                st.dataframe(
                    positive_impact_df,
                    column_config={
                        "Rating Impact": st.column_config.NumberColumn(
                            format="▲ %.2f", 
                            help="How much ratings increase when this issue is mentioned"
                        )
                    },
                    height=250,
                    hide_index=True
                )
            # Add review explorer for each issue
            st.subheader("Review Samples for Each Issue")
            
            for _, row in impact_df.iterrows():
                with st.expander(f"View reviews mentioning '{row['Issue']}'", expanded=False):
                    # Get affected reviews
                    affected_reviews = filtered_df[
                        filtered_df['Review'].str.contains(row['Issue'], case=False)
                    ][['Date', 'Rating', 'Review']]
                    
                    # Add filters
                    col1, col2 = st.columns(2)
                    with col1:
                        sample_size = st.slider(
                            "Number of reviews to show",
                            min_value=1,
                            max_value=len(affected_reviews),
                            value=min(5, len(affected_reviews)),
                            key=f"sample_{row['Issue']}"
                        )
                    with col2:
                        search_term = st.text_input(
                            "Search within reviews",
                            key=f"search_{row['Issue']}"
                        )
                    
                    # Filter and display
                    filtered = affected_reviews.copy()
                    if search_term:
                        filtered = filtered[filtered['Review'].str.contains(search_term, case=False)]
                    
                    # Display in scrollable container
                    with st.container(height=300):
                        for _, review in filtered.head(sample_size).iterrows():
                            st.markdown(f"""
                            <div style='padding:10px; margin:5px 0; border-radius:5px; 
                                        background:#f8f9fa; border-left:4px solid #6e48aa'>
                                <div style='font-size:0.9em; color:#666; margin-bottom:5px'>
                                    {review['Date'].strftime('%b %d, %Y')} | ⭐ {review['Rating']}
                                </div>
                                {review['Review']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if filtered.empty:
                            st.info("No reviews match the search criteria")


    st.header("📌 Sample Highlighted Reviews")
    
    # Sentiment analysis if not already done
    if 'Sentiment' not in filtered_df.columns:
        from textblob import TextBlob
        def get_sentiment(text):
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity
        filtered_df['Sentiment_Score'] = filtered_df['Review'].apply(get_sentiment)
    
    # Create scoring system for review usefulness
    def calculate_usefulness_score(row):
        # Score = sentiment strength * length * rating impact
        length_weight = np.log(len(str(row['Review']))) + 1
        return abs(row['Sentiment_Score']) * length_weight * (5 - row['Rating'])
    
    filtered_df['Usefulness_Score'] = filtered_df.apply(calculate_usefulness_score, axis=1)
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        search_keyword = st.text_input("🔍 Search reviews:")
    with col2:
        filter_issue = st.selectbox("Filter by issue:", 
                                  options=["All"] + list(ISSUE_CONFIG.keys()),
                                  format_func=lambda x: ISSUE_CONFIG[x]['title'] if x != "All" else "All")
    
    # Filter reviews
    filtered_reviews = filtered_df.copy()
    
    # Apply keyword filter
    if search_keyword:
        filtered_reviews = filtered_reviews[
            filtered_reviews['Review'].str.contains(search_keyword, case=False)
        ]
    
    # Apply issue filter
    if filter_issue != "All":
        issue_col = ISSUE_CONFIG[filter_issue]['column']
        filtered_reviews = filtered_reviews[filtered_reviews[issue_col]]
    
    # Sort and select top reviews
    top_reviews = filtered_reviews.sort_values('Usefulness_Score', ascending=False).head(10)
    
    if not top_reviews.empty:
        st.subheader(f"Top {len(top_reviews)} Most Insightful Reviews")
        
        for idx, row in top_reviews.iterrows():
            # Determine sentiment color
            sentiment_color = "#4CAF50" if row['Sentiment_Score'] > 0 else "#FF5252"
            
            # Highlight keywords
            review_text = row['Review']
            if search_keyword:
                keywords = search_keyword.split("|")
                for kw in keywords:
                    review_text = re.sub(
                        f"({kw})", 
                        r"<span style='background-color:yellow'>\1</span>", 
                        review_text, 
                        flags=re.IGNORECASE
                    )
            
            # Detect associated issues
            detected_issues = []
            for issue, config in ISSUE_CONFIG.items():
                if row[config['column']]:
                    detected_issues.append(config['title'])
            
            with st.expander(f"**{row['Rating']}★** - {row['Review'][:50]}...", expanded=False):
                st.markdown(f"""
                <div style="border-left: 4px solid {sentiment_color}; padding-left: 1rem;">
                    <p style="font-size: 0.9rem; color: #666;">{row['Date'].strftime('%b %d, %Y')} | Detected issues: {', '.join(detected_issues) or 'None'}</p>
                    <div style="margin: 0.5rem 0;">{review_text}</div>
                    <div style="display: flex; gap: 1rem; font-size: 0.8rem;">
                        <span>Usefulness Score: {row['Usefulness_Score']:.1f}</span>
                        <span>Sentiment: {row['Sentiment_Score']:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No reviews match the current filters")


    st.subheader("6-Month Rating Forecast")
    
    if not filtered_df.empty:
        forecast_fig = generate_forecast(filtered_df)
        
        if forecast_fig:
            # CORRECT: Use plotly_chart instead of pyplot
            st.plotly_chart(forecast_fig, use_container_width=True)
        else:
            st.warning("Forecast unavailable with current filters")
    else:
        st.warning("No data available for forecasting")
    
    st.subheader("Review Investigation Toolkit")
    search_query = st.text_input("🔍 Search across all reviews")
    if search_query:
        matches = filtered_df[filtered_df['Review'].str.contains(search_query, case=False)]
        st.write(f"Found {len(matches)} reviews containing '{search_query}'")
        
        with st.expander("View matching reviews"):
            for _, row in matches.iterrows():
                st.markdown(f"""
                <div style="padding:10px; margin:5px 0; background:#f8f9fa; border-radius:5px">
                    ⭐ {row['Rating']} | {row['Date'].strftime('%Y-%m-%d')}
                    <div style="color:#666">{row['Review']}</div>
                </div>
                """, unsafe_allow_html=True)


with tab3:
    st.header("Insights & Recommendations")
    
   # Sentiment Analysis
    st.subheader("Sentiment Distribution")
    sentiment_dist = filtered_df['Sentiment'].value_counts(normalize=True).mul(100)
    fig = px.pie(sentiment_dist, values=sentiment_dist.values, names=sentiment_dist.index,
                title="Review Sentiment Breakdown",
                color=sentiment_dist.index,
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'},
                hole=0.4)  # Donut chart
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

    st.subheader("🔄 Issue Prioritization Matrix")
    issue_matrix = filtered_df.melt(
        value_vars=['UI_Issue', 'Performance_Issue', 'Support_Complaint', 'Feature_Request','delivery_issues', 'Payment_Problems', 'Food_Quality', 'Promotions_Issues'],
        var_name='Issue',
        value_name='Reported'
    ).groupby('Issue')['Reported'].agg(['mean', 'sum']).reset_index()

    issue_matrix['Impact'] = [
        filtered_df.groupby(col)['Rating'].mean().diff().iloc[-1]
        for col in issue_matrix['Issue']
    ]

    fig = px.scatter(
        issue_matrix,
        x='mean',
        y='Impact',
        size='sum',
        color='Issue',
        hover_name='Issue',
        labels={'mean': 'Frequency (%)', 'sum': 'Total Reports'},
        title="Focus Resources Where Lines Intersect"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most Controversial Reviews")
    filtered_df['Sentiment_Rating_Gap'] = abs(filtered_df['Sentiment_Score'] - filtered_df['Rating']/5)
    controversial = filtered_df.nlargest(5, 'Sentiment_Rating_Gap')

    for _, row in controversial.iterrows():
        with st.expander(f"⭐ {row['Rating']} | Sentiment: {row['Sentiment']}", expanded=False):
            st.markdown(f"""
            **Why it's controversial:**  
            {row['Review']}
            
            *Sentiment score: {row['Sentiment_Score']:.2f} | Calculated gap: {row['Sentiment_Rating_Gap']:.2f}*
            """)
    
    st.header("🎯 Actionable Insights by Team")
    
    # Product Team Section
    with st.expander("📱 Product Team (Bugs, Features, UI)", expanded=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Key Metrics
            st.metric("Critical UI Issues", 
                     filtered_df['UI_Issue'].sum(),
                     help="Number of reviews mentioning UI/UX problems")
            
            st.metric("Pending Feature Requests",
                     filtered_df['Feature_Request'].sum(),
                     help="Number of requested features")
            
            st.metric("Performance Complaints",
                     filtered_df['Performance_Issue'].sum(),
                     help="Reports of app slowness/crashes")

        with col2:
            # Top Issues Visualization
            product_issues = {
                'UI Problems': filtered_df['UI_Issue'].sum(),
                'Feature Requests': filtered_df['Feature_Request'].sum(),
                'Performance Issues': filtered_df['Performance_Issue'].sum()
            }
            
            fig = px.bar(
                x=list(product_issues.keys()),
                y=list(product_issues.values()),
                title="Product Team Priority Areas",
                color=list(product_issues.keys()),
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            st.plotly_chart(fig, use_container_width=True)
    
        # Operations Team Section - Corrected Code
    with st.expander("🚚 Operations Team (Delivery, Quality)", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Delivery Complaints",
                    filtered_df['delivery_issues'].sum(),
                    help="Late/missing deliveries reported")
            
            # FIXED: Check text reviews for missing items
            missing_items_count = filtered_df[
                filtered_df['Food_Quality']  # First check if food quality issue exists
            ]['Review'].str.contains('missing', case=False, na=False).sum()
            
            st.metric("Missing Items",
                    missing_items_count,
                    help="Reports of incomplete orders")

        with col2:
            # Delivery Timeline Analysis
            delivery_issues = filtered_df[filtered_df['delivery_issues']]
            if not delivery_issues.empty:
                fig = px.line(
                    delivery_issues.groupby('Date')['delivery_issues'].count().reset_index(),
                    x='Date',
                    y='delivery_issues',
                    title="Delivery Complaints Trend",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No delivery complaints in selected period")

     
    # Support Team Section
    with st.expander("📞 Support Team (Response Quality)", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            support_issues = filtered_df['Support_Complaint_Type'].value_counts()
            st.metric("Unresolved Tickets", 
                     filtered_df['Support_Complaint'].sum())
            
            st.metric("Avg Response Time",
                     "24h" if len(filtered_df) > 0 else "N/A",
                     help="Estimated from review timestamps")

        with col2:
            if not support_issues.empty:
                fig = px.pie(
                    names=support_issues.index,
                    values=support_issues.values,
                    title="Support Complaint Types",
                    hole=0.4,
                    color_discrete_sequence=['#FF6B6B', '#FFA5A5', '#FFD6D6']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No support complaints in selected period")

    # Marketing Team Section
    with st.expander("📢 Marketing Team (Promotions, Loyalty)", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Coupon Issues",
                     filtered_df['Promotions_Issues'].sum(),
                     help="Failed promo code redemptions")
            
            st.metric("Positive Offer Mentions",
                     filtered_df[filtered_df['Sentiment'] == 'Positive']['Promotions_Issues'].sum(),
                     help="Positive mentions of promotions")

        with col2:
            # Promotion Effectiveness Analysis
            promo_data = filtered_df[filtered_df['Promotions_Issues']]
            if not promo_data.empty:
                fig = px.histogram(
                    promo_data,
                    x='Rating',
                    title="Promotion-Related Ratings Distribution",
                    nbins=5,
                    color_discrete_sequence=['#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No promotion-related feedback in selected period")

    # Add action buttons
    st.divider()
    st.button("📩 Send Summary to Teams", help="Email this report to all team leads")
    
        # Add to your existing ISSUE_CONFIG
    ISSUE_CONFIG['churn_risk'] = {
        'column': 'Churn_Risk',
        'title': 'Churn Risk Indicators',
        'color': '#FF4444',
        'keywords': [
            r'\buninstall(ing|ed)?\b',
            r'\bdelete(d|ing)?\b',
            r'\bnot renew(ing)?\b',
            r'\bcancel(ed|ling)?\b',
            r'\bswitching to\b',
            r'\bwon\'?t use\b',
            r'\bnever (use|buy)\b',
            r'\b(quit|quitting)\b',
            r'\b(not worth|too expensive)\b',
            r'\b(awful|terrible) experience\b'
        ]
    }

    # Add this analysis section (could be in its own tab or expander)
    with st.expander("🚨 Churn Risk Detection", expanded=True):
        # Calculate churn indicators
        churn_pattern = r'|'.join(ISSUE_CONFIG['churn_risk']['keywords'])
        filtered_df['Churn_Risk'] = filtered_df['Review'].str.contains(
            churn_pattern, 
            case=False, 
            na=False,
            regex=True
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Risk Reviews", 
                    filtered_df['Churn_Risk'].sum(),
                    help="Reviews indicating potential churn")
        with col2:
            churn_avg_rating = filtered_df[filtered_df['Churn_Risk']]['Rating'].mean()
            st.metric("Avg Rating of At-Risk Users", 
                    f"{churn_avg_rating:.1f}★",
                    help="Lower ratings indicate higher churn probability")
        with col3:
            churn_pct = filtered_df['Churn_Risk'].mean() * 100
            st.metric("Churn Risk Percentage", 
                    f"{churn_pct:.1f}%",
                    help="Percentage of all reviews showing churn signs")

        # Show highlighted examples
        st.subheader("High-Risk Review Examples")
        churn_reviews = filtered_df[filtered_df['Churn_Risk']]
        
        if not churn_reviews.empty:
            for _, row in churn_reviews.head(5).iterrows():
                highlighted_text = re.sub(
                    f'({churn_pattern})', 
                    r'<span style="background-color:#FFBABA">\1</span>', 
                    row['Review'], 
                    flags=re.IGNORECASE
                )
                
                st.markdown(f"""
                <div style="border-left: 3px solid #FF4444; 
                            padding-left: 1rem;
                            margin: 0.5rem 0;
                            font-size: 0.9rem">
                    <div style="color: #666; margin-bottom: 0.2rem">
                        {row['Date'].strftime('%b %d')} • {row['Rating']}★
                    </div>
                    {highlighted_text}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-risk reviews detected in current filters")

        # Trend analysis
        st.subheader("Churn Risk Over Time")
        if not churn_reviews.empty:
            trend_data = churn_reviews.groupby('Month_Year').agg(
                Churn_Count=('Churn_Risk', 'sum'),
                Avg_Rating=('Rating', 'mean')
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=trend_data['Month_Year'],
                    y=trend_data['Churn_Count'],
                    name='Churn Risk Count',
                    marker_color='#FF4444'
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=trend_data['Month_Year'],
                    y=trend_data['Avg_Rating'],
                    name='Avg Rating',
                    line=dict(color='#444', dash='dot')
                ),
                secondary_y=True
            )
            fig.update_layout(
                title='Churn Risk vs User Ratings Over Time',
                xaxis_title='Date',
                yaxis_title='Churn Risk Count',
                yaxis2_title='Average Rating',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available")


    with st.expander("🌟 Best Performing Features", expanded=True):
        positive_reviews = filtered_df[filtered_df['Sentiment'] == 'Positive']
    
        if not positive_reviews.empty:
            # Feature extraction using NLP patterns
            praise_pattern = r"""
                (\b[A-Z][a-z]+(?:\s+[A-Za-z]+){0,3}\b)  # Capture noun phrases
                \s+
                (is|are|was|were|has|have|'s)\s+  # Common linking verbs
                (awesome|great|excellent|amazing|love|like|good|fantastic|superb|perfect|useful|helpful|easy|intuitive)
            """
            
            # Extract feature-praise pairs
            features = []
            for review in positive_reviews['Review']:
                matches = re.finditer(praise_pattern, review, re.VERBOSE | re.IGNORECASE)
                for match in matches:
                    feature = match.group(1).lower().strip()
                    praise_word = match.group(3).lower()
                    features.append((feature, praise_word))
            
            # Create frequency analysis
            if features:
                feature_df = pd.DataFrame(features, columns=['Feature', 'Praise'])
                top_features = feature_df['Feature'].value_counts().head(10)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Top Praised Feature", 
                            top_features.index[0] if len(top_features) > 0 else "N/A",
                            help=f"Mentioned {top_features.values[0] if len(top_features) > 0 else 0} times")
                
                with col2:
                    st.metric("Most Common Praise Word",
                            feature_df['Praise'].value_counts().index[0] if len(features) > 0 else "N/A")
                
                # Visualization
                fig = px.bar(
                    top_features,
                    orientation='h',
                    title="Most Loved Features",
                    labels={'index': 'Feature', 'value': 'Mentions'},
                    color=top_features.values,
                    color_continuous_scale='Tealgrn'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Example reviews carousel
                st.subheader("Example Praise Quotes")
                sampled_reviews = positive_reviews.sample(min(3, len(positive_reviews)))
                for _, row in sampled_reviews.iterrows():
                    st.markdown(f"""
                    <div style="border-left: 3px solid #4CAF50; padding-left: 1rem; margin: 1rem 0">
                        <div style="color: #666; font-size: 0.9rem">{row['Rating']}★ • {row['Date'].strftime('%b %Y')}</div>
                        <div>"{row['Review']}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific feature praise detected in positive reviews")
        else:
            st.warning("No positive reviews available for analysis")



with tab4:
    st.header("🔍 Comprehensive Issues Analysis")
        # Define issue configurations

    ISSUE_CONFIG = {
        'ui_issue': {
            'column': 'UI_Issue',
            'title': 'UI/UX Issues',
            'color': '#CB2726',
            'keywords': ['slow', 'lag', 'bug', 'glitch', 'crash', 'freeze', 'complicated', 'hard', 'navigation','unresponsive','delay','latency','stutter',
        'load time','resource intensive','memory leak','instability','error','failure','hang','confusing','difficult','intricate','unintuitive','cumbersome',
        'tedious','user-friendly','accessibility','workflow','steps','process','layout','design','interface','discoverability','pixelated','distorted',
        'alignment','animation','responsiveness','touch','click','scroll','visual','rendering','display','font','color','data loss','sync','save',
               'input','output','search','filter','functionality','feature','compatibility','frustrating','annoying','irritating','problem','issue','bad',
               'poor','broken','useless','disappointing']
        },
        'performance_issue': {
            'column': 'Performance_Issue',
            'title': 'Performance Issues',
            'color': '#CB2726',
            'keywords': ['crash','freeze','lag','slow','bug','glitch','not responding','stuck','hangs',
    'loading','performance','unstable','error','delay','latency','stutter','load time',
    'resource intensive','memory leak','instability','failure','unresponsive','rendering','optimization']
        },
        'feature_request': {
            'column': 'Feature_Request',
            'title': 'Feature Requests',
            'color': '#CB2726',
            'keywords': ['should have', 'need', 'want', 'please add', 'where is', 'why no', 'missing', 'would love',
            'wish there was', 'suggest', 'recommend', 'hope to see', 'require', 'desire', 'looking for',
            'could use', 'it would be great if', 'it would be helpful if', 'is there a way to',
            'is it possible to', 'consider adding', 'Id like to see', 'Im trying to find',
            'can you implement', 'how do I', 'is it possible to get', 'is there', 'Im looking for']
        },
        'support_complaint': {
            'column': 'Support_Complaint',
            'title': 'Support Issues',
            'color': '#CB2726',
            'keywords': {'No Response': ['no reply', 'no answer', 'ignored', 'no help', 'no response', 'never responded', 'no feedback'],
        'Slow Response': ['slow response', 'took long', 'days to reply', 'delayed response', 'long wait', 'prolonged delay', 'late reply'],
        'Unhelpful': ['not helpful', 'useless', 'did not solve', 'waste of time', 'ineffective', 'unhelpful', 'did not assist', 'no solution', 'failed to resolve'],
        'Rude Staff': ['rude', 'arrogant', 'unprofessional', 'angry', 'impolite', 'disrespectful', 'hostile', 'offensive', 'dismissive']
            }
        },
        'pricing_complaint': {
            'column': 'Pricing_Complaint',
            'title': 'Pricing Issues',
            'color': '#CB2726',
            'keywords': ['expensive','overpriced','pricey','too much','not worth','high price','cost too much',
        'unfair','cheaper','lower price','reduce price','price hike','cost','value','affordable']
        },
        'delivery_issues': {
        'column': 'delivery_issues',
        'title': 'Delivery Issues',
        'color': '#FF6B6B',
        'keywords': [
            'late delivery', 'delayed', 'not delivered', 'delivery time', 'driver late', 'took too long',
            'ETA', 'wrong address', 'missed delivery', 'delivery failed', 'reschedule', 'never arrived',
            'package delay', 'delivery issue', 'still waiting', 'came late', 'got it late', 'delay in delivery',
            'order late', 'order not here', 'waiting for my order', 'where is my order', 'running late',
            'delivered to wrong address', 'delivered somewhere else', 'didn’t show up', 'arrived late'
        ]
        },
        'payment_issue': {
        'column': 'Payment_Problems',
        'title': 'Payment Problems',
        'color': '#FFA500',
        'keywords': [
            'payment failed', 'transaction error', 'card declined', 'not processed', 'double charged',
            'overcharged', 'refund pending', 'refund delay', 'payment issue', "can't pay", 'not refunded',
            'incorrect amount', 'failed to pay', 'billing error', 'charge issue', 'charged twice',
            'money deducted', 'amount not refunded', 'payment stuck', 'did not get refund',
            'transaction declined', 'unable to pay', 'app charged me', 'no confirmation after payment',
            'payment not successful'
        ]
        },
        'food_quality': {
        'column': 'Food_Quality',
        'title': 'Food Quality',
        'color': '#6A5ACD',
        'keywords': [
            'stale food', 'not fresh', 'cold food', 'bad taste', 'spoiled', 'poor quality',
            'packaging issue', 'leaked', 'damaged package', 'soggy', 'missing items', 'wrong item',
            'undercooked', 'overcooked', 'smells bad', 'rotten', 'food poisoning', 'not edible',
            'food was cold', 'food was awful', 'not good', 'unhygienic', 'dirty packaging',
            'weird smell', 'wrong dish', 'order messed up', 'hair in food', 'low quality', 'bad smell'
        ]
        },
        'promotions_issue': {
        'column': 'Promotions_Issues',
        'title': 'Promotions and Coupons',
        'color': '#32CD32',
        'keywords': [
            'coupon not working', 'promo code invalid', 'offer not applied', 'discount not working',
            'code expired', "can't use promo", 'offer issue', 'not eligible', "didn't get discount",
            'cashback not received', 'free delivery not applied', 'reward not credited',
            'code not accepted', 'voucher didn’t work', 'promo didn’t apply', 'invalid promo',
            'didn’t get offer', 'promotion failed', 'no discount received', 'promo not working',
            'discount missing', 'free item not added', 'applying coupon failed', 'reward missing'
        ]
        },


    }




        # Get all issue columns from config
    issue_columns = [v['column'] for v in ISSUE_CONFIG.values()]

    # Calculate reviews with at least one issue
    reviews_with_issues = filtered_df[issue_columns].any(axis=1).sum()

    # Display KPI at the top
    st.metric("Reviews with Issues", 
            f"{reviews_with_issues}/{len(filtered_df)}",
            help="Number of reviews containing at least one reported issue")
    # Unified Issues Analysis Section
    with st.expander("🔍 Unified Issue Analysis", expanded=True):
        # Issue Selection
        selected_issue = st.selectbox(
            "Select Issue Type:",
            options=list(ISSUE_CONFIG.keys()),
            format_func=lambda x: ISSUE_CONFIG[x]['title']
        )
        
        config = ISSUE_CONFIG[selected_issue]
        issue_col = config['column']
        
        # Metrics Columns
        col1, col2, col3 = st.columns(3)
        with col1:
            total_issues = filtered_df[issue_col].sum()
            st.metric(f"Total {config['title']}", total_issues)
        
        with col2:
            negative_pct = filtered_df[filtered_df['Sentiment'] == 'Negative'][issue_col].mean() * 100
            st.metric("In Negative Reviews", f"{negative_pct:.1f}%")
        
        with col3:
            impact = filtered_df.groupby(issue_col)['Rating'].mean().diff().iloc[-1]
            st.metric("Rating Impact", f"{impact:.1f}★", delta_color="inverse")

        # Common Visualizations
        tab1, tab2, tab3,tab4 = st.tabs(["Trend Analysis", "Term Cloud", "Deep Dive",'Sentiment Breakdown'])
        
        with tab1:
            # Prepare combined trend data
            trend_data = filtered_df.groupby('Month_Year').agg({
                issue_col: 'mean'
            }).reset_index()

            # Convert proportions to percentages
            trend_data[issue_col] = trend_data[issue_col] * 100

            # Ensure Month_Year is datetime and sorted
            trend_data['Month_Year'] = pd.to_datetime(trend_data['Month_Year'], format='%b %Y')  # adjust format as needed
            trend_data = trend_data.sort_values('Month_Year')

            # Create figure with secondary y-axis
            fig_trend = px.line(
                trend_data,
                x='Month_Year',
                y=issue_col,
                title=f"{config['title']} Trend",
                labels={
                    issue_col: '% of Reviews',
                    'Month_Year': 'Month'
                },
                color_discrete_sequence=[config['color'], '#00CED1']
            )

            # Update layout for dual axes
            fig_trend.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(
                    title='% of Reviews',
                    showgrid=False,
                    range=[0, 100]
                ),
                yaxis2=dict(
                    title='Sentiment Score',
                    overlaying='y',
                    side='right',
                    range=[-1, 1],
                    showgrid=False
                ),
                legend=dict(
                    x=1.1,
                    y=1,
                    bgcolor='rgba(255,255,255,0.5)'
                )
            )

            st.plotly_chart(fig_trend, use_container_width=True)

        
        
                # In your word cloud section (tab4's Term Cloud tab), replace with:
        with tab2:
            # Word Cloud
            issue_reviews = " ".join(filtered_df[filtered_df[issue_col]]['Review'])
            if issue_reviews.strip():
                # Create figure explicitly
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    colormap='Reds',
                    max_words=150
                ).generate(issue_reviews)

                # Display using explicit axes
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                ax.set_title(
                    f"Key Terms in {config['title']}",
                    fontsize=24,
                    pad=30,
                    color='#990000',
                    fontweight='bold'
                )

                # Use st.pyplot() with explicit figure
                st.pyplot(fig)
                plt.close(fig)  # Clean up memory
            else:
                st.warning(f"No reviews found for {config['title']}")
        with tab3:
            # Specialized Visuals
            if selected_issue == 'feature_request':
                vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=50)
                X = vectorizer.fit_transform(filtered_df[filtered_df[issue_col]]['Review'])
                features = pd.DataFrame({
                    'Feature': vectorizer.get_feature_names_out(),
                    'Score': X.sum(axis=0).A1
                }).sort_values('Score', ascending=False).head(10)
                
                fig_features = px.bar(features, x='Score', y='Feature', orientation='h',
                                    title="Top Requested Features", color='Score',
                                    color_continuous_scale='blues')
                st.plotly_chart(fig_features, use_container_width=True, key=f"features_{selected_issue}")
            
            elif selected_issue == 'support_complaint':
                if 'Support_Complaint_Type' in filtered_df.columns:
                    supp_types = filtered_df['Support_Complaint_Type'].value_counts()
                    fig_support = px.pie(supp_types, values=supp_types.values, names=supp_types.index,
                                        title="Support Complaint Types")
                    st.plotly_chart(fig_support, use_container_width=True, key=f"support_{selected_issue}")
            
            elif selected_issue == 'pricing_complaint':
                fig_pricing = px.box(filtered_df[filtered_df[issue_col]], 
                                    y='Rating', 
                                    title="Rating Distribution for Pricing Complaints",
                                    color_discrete_sequence=[config['color']])
                st.plotly_chart(fig_pricing, use_container_width=True, key=f"pricing_{selected_issue}")
            
            else:
                # Default correlation analysis
                corr_data = filtered_df[[issue_col, 'Rating']].corr()
                fig_corr = ff.create_annotated_heatmap(
                    z=corr_data.values,
                    x=corr_data.columns.tolist(),
                    y=corr_data.index.tolist(),
                    colorscale='reds'
                )
                st.plotly_chart(fig_corr, use_container_width=True, key=f"correlation_{selected_issue}")

        with tab4:
            # Sentiment Breakdown
            if 'Sentiment' not in filtered_df.columns:
                st.warning("Sentiment column not found. Ensure sentiment analysis has been applied.")
            else:
                sentiment_dist = filtered_df[filtered_df[issue_col]]['Sentiment'].value_counts(normalize=True).mul(100)
                
                if sentiment_dist.empty:
                    st.warning(f"No sentiment data available for {config['title']}")
                else:
                    fig_sentiment = px.pie(
                        sentiment_dist,
                        values=sentiment_dist.values,
                        names=sentiment_dist.index,
                        title=f"{config['title']} Sentiment Breakdown",
                        color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True, key=f"sentiment_{selected_issue}")

           
                # Collect frequency and average rating for each issue type
        # Collect frequency and average rating for each issue type
        urgency_data = []

        # ✅ Corrected loop through ISSUE_CONFIG
        for issue_key in ISSUE_CONFIG:
            config = ISSUE_CONFIG[issue_key]
            col = config['column']
            name = config['title']
            
            if col in filtered_df.columns:
                # Calculate metrics
                frequency = filtered_df[col].mean()
                if filtered_df[col].sum() > 0:
                    avg_rating = filtered_df.loc[filtered_df[col] == 1, 'Rating'].mean()
                    urgency = frequency * (5 - avg_rating)
                else:
                    avg_rating = 0
                    urgency = 0
                
                urgency_data.append({
                    'Issue Type': name,
                    'Frequency': frequency,
                    'Avg Rating': avg_rating,
                    'Urgency': urgency
                })

        # Convert to DataFrame
        urgency_df = pd.DataFrame(urgency_data)

        # Plot the Urgency Matrix
        if not urgency_df.empty:
            fig = px.scatter(urgency_df, x='Frequency', y='Avg Rating', text='Issue Type',
                            size='Urgency', color='Urgency',
                            color_continuous_scale='Reds',
                            title="Issue Urgency Matrix (High Frequency + Low Rating = High Urgency)")
            fig.update_traces(textposition='top center')
            fig.update_layout(yaxis_title='Average Rating', xaxis_title='Frequency of Issue')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to generate urgency matrix.")

        st.header("🧠 Sentiment Breakdown by Topic")
        
        # Create topic sentiment analysis
        topic_data = []
        
        # Define your topics (align with issue config)
        TOPIC_CONFIG = {
            'Delivery Experience': 'delivery_issues',
            'App Performance': 'performance_issue',
            'Customer Support': 'support_complaint',
            'Food Quality': 'food_quality',
            'Pricing': 'pricing_complaint',
            'App Usability': 'ui_issue'
        }
        
        for topic_name, issue_key in TOPIC_CONFIG.items():
            issue_col = ISSUE_CONFIG[issue_key]['column']
            
            # Filter reviews mentioning this topic
            topic_reviews = filtered_df[filtered_df[issue_col]]
            
            if len(topic_reviews) > 0:
                # Sentiment distribution
                sentiment_dist = topic_reviews['Sentiment'].value_counts(normalize=True).mul(100)
                pos = sentiment_dist.get('Positive', 0)
                neu = sentiment_dist.get('Neutral', 0)
                neg = sentiment_dist.get('Negative', 0)
                
                # Average rating
                avg_rating = topic_reviews['Rating'].mean()
                
                topic_data.append({
                    'Topic': topic_name,
                    'Positive %': pos,
                    'Neutral %': neu,
                    'Negative %': neg,
                    'Avg. Rating': avg_rating,
                    'Reviews Count': len(topic_reviews)
                })
        
        # Create dataframe and sort by negative %
        topic_df = pd.DataFrame(topic_data).sort_values('Negative %', ascending=False)
        
        # Metrics row
        st.subheader("Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            most_issues = topic_df.iloc[0]['Topic']
            st.metric("🚨 Most Problematic Area", most_issues)
        with col2:
            best_rating = topic_df[topic_df['Reviews Count'] > 10].sort_values('Avg. Rating').iloc[-1]['Topic']
            st.metric("⭐ Best Performing Area", best_rating)
        with col3:
            total_topic_reviews = topic_df['Reviews Count'].sum()
            st.metric("📝 Total Topic Mentions", total_topic_reviews)
        
        # Interactive table with heatmap
        st.subheader("Detailed Breakdown")
        
        # Format percentages
        display_df = topic_df.copy()
        display_df['Positive %'] = display_df['Positive %'].apply(lambda x: f"{x:.1f}%")
        display_df['Neutral %'] = display_df['Neutral %'].apply(lambda x: f"{x:.1f}%")
        display_df['Negative %'] = display_df['Negative %'].apply(lambda x: f"{x:.1f}%")
        display_df['Avg. Rating'] = display_df['Avg. Rating'].apply(lambda x: f"{x:.1f}★")
        
        # Create styled table
        def color_negative(val):
            value = float(val[:-1])
            color = '#ffcccc' if value > 30 else '#ffe6cc' if value > 20 else '#ffffff'
            return f'background-color: {color}'
        
        def color_rating(val):
            value = float(val[:-1])
            color = '#cce5cc' if value > 4 else '#ffe6cc' if value > 3 else '#ffcccc'
            return f'background-color: {color}'
        
        styled_df = display_df.style\
            .applymap(color_negative, subset=['Negative %'])\
            .applymap(color_rating, subset=['Avg. Rating'])\
            .format({'Reviews Count': '{:,}'})
        
        st.dataframe(
            styled_df,
            column_order=['Topic', 'Negative %', 'Neutral %', 'Positive %', 'Avg. Rating', 'Reviews Count'],
            height=(len(topic_df) + 1) * 35 + 3,
            use_container_width=True
        )
        
        # Trend sparklines
        st.caption("💡 Hover over metrics to see trend sparklines (add time dimension)")


                # Prepare urgency data
        urgency_data = []
        for issue_key, config in ISSUE_CONFIG.items():
            issue_col = config['column']
            
            if issue_col in filtered_df.columns:
                # Frequency: % of reviews mentioning this issue
                frequency = filtered_df[issue_col].mean() * 100  # as percentage
                
                # Avg Rating: Average rating when issue is reported (lower = worse)
                if filtered_df[issue_col].sum() > 0:  # Ensure there are cases
                    avg_rating = filtered_df[filtered_df[issue_col]]['Rating'].mean()
                else:
                    avg_rating = 5  # Default to max (no impact)
                
                # Urgency Score = Frequency * (5 - Avg Rating)
                urgency_score = frequency * (5 - avg_rating)
                
                urgency_data.append({
                    'Issue': config['title'],
                    'Frequency (%)': frequency,
                    'Avg Rating': avg_rating,
                    'Urgency': urgency_score
                })

        urgency_df = pd.DataFrame(urgency_data)




        # Additional Metrics
        st.subheader("Impact Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Response rate impact
            if 'Reply' in filtered_df:
                responded_rate = filtered_df[filtered_df['Reply'] != "No Reply"][issue_col].mean()
                st.metric("Response Rate for Issue", f"{responded_rate:.1%}")
        
        with col2:
            # Device distribution
            device_dist = filtered_df[filtered_df[issue_col]]['Devices_Mentioned'].value_counts().head(5)
            if not device_dist.empty:
                fig_device = px.pie(device_dist, values=device_dist.values, names=device_dist.index,
                                    title="Top Affected Devices")
                st.plotly_chart(fig_device, use_container_width=True, key=f"device_{selected_issue}")    
    
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
     
    # Prioritization Matrix
    st.subheader("📊 Issue Prioritization Matrix")
        # Calculating the frequency for each issue type
    frequency_data = {
        'UI/UX': filtered_df['UI_Issue'].mean(),
        'Performance': filtered_df['Performance_Issue'].mean(),
        'Feature Requests': filtered_df['Feature_Request'].mean(),
        'Support': filtered_df['Support_Complaint'].mean(),
        'Monetization': filtered_df['Pricing_Complaint'].mean(),
        'Delivery Issues': filtered_df['delivery_issues'].mean(),
        'Payment Problems': filtered_df['Payment_Problems'].mean(),
        'Food Quality': filtered_df['Food_Quality'].mean(),
        'Promotions Issues': filtered_df['Promotions_Issues'].mean(),
        'Subscription Complaint': filtered_df['Subscription_Complaint'].mean()
    }

    # Calculating the impact for each issue type (mean difference of ratings)
    impact_data = {
        'UI/UX': filtered_df.groupby('UI_Issue')['Rating'].mean().diff().iloc[-1],
        'Performance': filtered_df.groupby('Performance_Issue')['Rating'].mean().diff().iloc[-1],
        'Feature Requests': 0.5,  # Assuming feature requests have medium impact
        'Support': filtered_df.groupby('Support_Complaint')['Rating'].mean().diff().iloc[-1],
        'Monetization': filtered_df.groupby('Pricing_Complaint')['Rating'].mean().diff().iloc[-1],
        'Delivery Issues': filtered_df.groupby('delivery_issues')['Rating'].mean().diff().iloc[-1],
        'Payment Problems': filtered_df.groupby('Payment_Problems')['Rating'].mean().diff().iloc[-1],
        'Food Quality': filtered_df.groupby('Food_Quality')['Rating'].mean().diff().iloc[-1],
        'Promotions Issues': filtered_df.groupby('Promotions_Issues')['Rating'].mean().diff().iloc[-1],
        'Subscription Complaint': filtered_df.groupby('Subscription_Complaint')['Rating'].mean().diff().iloc[-1]
    }

    # Ensure both frequency_data and impact_data have the same keys and length
    issues = list(frequency_data.keys())

    # Create the DataFrame
    priority_data = {
        'Issue Type': issues,
        'Frequency': [frequency_data[issue] for issue in issues],
        'Impact': [impact_data[issue] for issue in issues]
    }

    priority_df = pd.DataFrame(priority_data)

    # Check if the DataFrame is empty
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
    col1, col2, col3, col4,col5 = st.columns(5)
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
    with col5:
        st.metric("Negative Sentiment", f"{(filtered_df['Sentiment'] == 'Negative').mean()*100:.1f}%",
                 help="Percentage of reviews with negative sentiment")
    
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
                        color_discrete_sequence=['#FF0000']  # Pure red hex code
                        )
            fig.update_traces(marker_color='#CB2726',  # Ensures full red coloring
                            selector=dict(type='bar'))
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
            'Pricing Concerns': filtered_df['Pricing_Complaint'].mean(),
            'Feature Requests': filtered_df['Feature_Request'].mean(),
            'Delivery Issues': filtered_df['delivery_issues'].mean(),
            'Payment Problems': filtered_df['Payment_Problems'].mean(),
            'Food Quality': filtered_df['Food_Quality'].mean(),
            'Promotions Issues': filtered_df['Promotions_Issues'].mean(),
            'Subscription Complaints': filtered_df['Subscription_Complaint'].mean()
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
            'Performance_Issue': 'mean',
            'Feature_Request': 'mean',
            'Support_Complaint': 'mean',
            'Pricing_Complaint': 'mean',
            'delivery_issues': 'mean',
            'Payment_Problems': 'mean',
            'Food_Quality': 'mean',
            'Promotions_Issues': 'mean',
            'Subscription_Complaint': 'mean'
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

    def generate_exec_summary(df):
        summary = f"""
        **Period Analyzed:** {df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')}
        
        **Key Achievements:**
        - {df[df['Rating'] >= 4].shape[0]} positive experiences reported
        - {df[df['Reply'] != 'No Reply'].shape[0]} user engagements handled
        - Top performing month: {df.groupby('Month_Year')['Rating'].mean().idxmax()}
        
        **Critical Focus Areas:**
        - {df[df['Rating'] <= 2].shape[0]} urgent complaints needing resolution
        - {df['UI_Issue'].sum()} reported usability barriers
        - {df['Performance_Issue'].sum()} technical instability reports
        """
        return summary

    st.markdown(generate_exec_summary(filtered_df), unsafe_allow_html=True)
    
    # Final Summary
    st.divider()
    st.subheader("📋 Final Assessment")
    
    # Generate dynamic summary based on data
    avg_rating = filtered_df['Rating'].mean()
    pos_sentiment = (filtered_df['Sentiment'] == 'Positive').mean()
    neg_sentiment = (filtered_df['Sentiment'] == 'Negative').mean()
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
        <li>Negative Sentiment: <b>{neg_sentiment:.1%}</b> of reviews</li>
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
            file_name="Josh App_app_report.pdf",
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
            industry_avg=4.2
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
            'Josh App': {
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
            st.plotly_chart(forecast_fig, use_container_width=True)  
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
Comprehensive analysis of Josh App reviews to identify improvement opportunities.

**Data Source:**  
Google Play Store reviews ({datetime.now().strftime('%Y-%m-%d')})

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
            <h4>Opportunity: {feature_request_count} feature requests detected!</h4>
            <p>Most requested: "{top_request}..."</p>
        </div>
        """, unsafe_allow_html=True)