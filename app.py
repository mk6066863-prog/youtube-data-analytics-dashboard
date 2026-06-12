from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os

# Matplotlib for data visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------------
# FLASK APPLICATION SETUP
# -----------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# -----------------------------------------
# HOME PAGE
# -----------------------------------------
@app.route('/')
def home():
    """
    Render the file upload page.
    """
    return render_template('index.html')


# -----------------------------------------
# ANALYSIS ROUTE
# -----------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():

    # -----------------------------------------
    # FILE UPLOAD
    # -----------------------------------------
    file = request.files['file']

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # -----------------------------------------
    # LOAD DATASET
    # -----------------------------------------
    df = pd.read_csv(filepath)

    # -----------------------------------------
    # DATA CLEANING
    # -----------------------------------------

    # Remove missing values
    df = df.dropna()

    # Convert publish_time to datetime format
    df['publish_time'] = pd.to_datetime(
        df['publish_time'],
        errors='coerce'
    )

    # Remove invalid dates
    df = df.dropna(subset=['publish_time'])

    # Extract upload day
    df['day'] = df['publish_time'].dt.day_name()

    # Remove rows with zero views
    df = df[df['views'] > 0]

    # -----------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------

    # Business Metric:
    # Engagement Rate shows audience interaction
    df['engagement_rate'] = (
        (df['likes'] + df['comment_count'])
        / df['views']
    ) * 100

    # -----------------------------------------
    # BUSINESS QUESTION 1:
    # Which categories generate
    # the highest audience engagement?
    # -----------------------------------------

    category_engagement = (
        df.groupby('category_id')['engagement_rate']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    # -----------------------------------------
    # BUSINESS QUESTION 2:
    # Which videos gained
    # the highest number of views?
    # -----------------------------------------

    top_videos = (
        df.sort_values(
            by='views',
            ascending=False
        )
        .head(10)
    )

    # -----------------------------------------
    # BUSINESS QUESTION 3:
    # Which day generates the
    # highest engagement?
    # -----------------------------------------

    best_day = (
        df.groupby('day')['engagement_rate']
        .mean()
        .idxmax()
    )
    # -----------------------------------------
    # KPI METRICS FOR DASHBOARD
    # -----------------------------------------

    avg_engagement = round(
    df['engagement_rate'].mean(),
    2
    )
    # Total dataset views
    total_views = int(
    df['views'].sum()
    )

    total_videos = len(df)
    # -----------------------------------------
    # BUSINESS QUESTION 4:
    # Which videos are viral?
    # Viral = Top 10% viewed videos
    # -----------------------------------------

    viral_threshold = df['views'].quantile(0.90)

    viral_videos = df[
        df['views'] >= viral_threshold
    ]

    # -----------------------------------------
    # CORRELATION ANALYSIS
    # -----------------------------------------

    correlation_matrix = df[
        [
            'views',
            'likes',
            'comment_count',
            'engagement_rate'
        ]
    ].corr()

    # -----------------------------------------
    # VISUALIZATION 1
    # Top Categories By Engagement
    # -----------------------------------------

    plt.figure(figsize=(10, 5))

    category_engagement.plot(
        kind='bar'
    )

    plt.title(
        'Top Categories by Engagement Rate'
    )

    plt.xlabel('Category ID')
    plt.ylabel('Engagement Rate (%)')

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            app.config['UPLOAD_FOLDER'],
            'category.png'
        )
    )

    plt.close()

    # -----------------------------------------
    # VISUALIZATION 2
    # Likes vs Views
    # -----------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df['likes'],
        df['views'],
        alpha=0.5
    )

    plt.title(
        'Likes vs Views'
    )

    plt.xlabel('Likes')
    plt.ylabel('Views')

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            app.config['UPLOAD_FOLDER'],
            'scatter.png'
        )
    )

    plt.close()

    # -----------------------------------------
    # VISUALIZATION 3
    # Best Upload Day Analysis
    # -----------------------------------------

    plt.figure(figsize=(10, 5))

    (
        df.groupby('day')['engagement_rate']
        .mean()
        .sort_values(ascending=False)
        .plot(kind='bar')
    )

    plt.title(
        'Average Engagement Rate by Upload Day'
    )

    plt.xlabel('Day')
    plt.ylabel('Engagement Rate (%)')

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            app.config['UPLOAD_FOLDER'],
            'day.png'
        )
    )

    plt.close()

    # -----------------------------------------
    # VISUALIZATION 4
    # Viral Videos
    # -----------------------------------------

    top_viral = (
        viral_videos
        .sort_values(
        by='views',
        ascending=False
    )
        .head(10)
    )

    # Shorten long titles for better graph readability
    top_viral['short_title'] = (
    top_viral['title']
    .astype(str)
    .str.slice(0, 35)
    )

    plt.figure(figsize=(12, 6))

    plt.barh(
    top_viral['short_title'],
    top_viral['views']
    )

    plt.title(
    'Top Viral Videos'
)

    plt.xlabel('Views')
    plt.ylabel('Video Title')

    plt.tight_layout()

    plt.savefig(
      os.path.join(
        app.config['UPLOAD_FOLDER'],
        'viral.png'
    )
    )

    plt.close()

    # -----------------------------------------
    # MACHINE LEARNING OBJECTIVE
    # Predict video views using
    # engagement metrics
    # -----------------------------------------

    X = df[
      [
        'likes',
        'comment_count',
        'engagement_rate'
      ]
    ]

    y = df['views']

    X_train, X_test, y_train, y_test = (
       train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
       )
    )

    model = LinearRegression()
    model.fit(
       X_train,
       y_train
    )

    # -----------------------------------------
    # MODEL EVALUATION
    # -----------------------------------------

    y_pred = model.predict(X_test)

    r2 = round(
       r2_score(y_test, y_pred) * 100,
       2
    )

    mae = int(
       mean_absolute_error(
        y_test,
        y_pred
       )
    )

    # -----------------------------------------
    # PREDICTION USING
    # AVERAGE DATASET VALUES
    # -----------------------------------------

    avg_likes = df['likes'].mean()
    avg_comments = df['comment_count'].mean()
    avg_eng_rate = df['engagement_rate'].mean()

    predicted_views = int(
        abs(
            model.predict(
              [[
                avg_likes,
                avg_comments,
                avg_eng_rate
              ]]
            )[0]
        )
    )

    # -----------------------------------------
    # ACTUAL VS PREDICTED GRAPH
    # -----------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
      y_test,
      y_pred,
      alpha=0.5
    )
    plt.plot(
      [y_test.min(), y_test.max()],
      [y_test.min(), y_test.max()],
      'r--'
    )

    plt.title(
      'Actual vs Predicted Views'
    )
    plt.xlabel('Actual Views')
    plt.ylabel('Predicted Views')

    plt.tight_layout()

    plt.savefig(
      os.path.join(
        app.config['UPLOAD_FOLDER'],
        'prediction.png'
      )
    )

    plt.close()

    # -----------------------------------------
    # CONVERT RESULTS TO HTML
    # -----------------------------------------

    category_table = (
        category_engagement
        .to_frame(
            name='Avg Engagement Rate (%)'
        )
        .to_html(
            classes='table table-striped'
        )
    )

    video_table = (
        top_videos[
            ['title', 'views']
        ]
        .to_html(
            classes='table table-striped',
            index=False
        )
    )

    # -----------------------------------------
    # SEND RESULTS TO TEMPLATE
    # -----------------------------------------

    return render_template(
    'result.html',
    category_table=category_table,
    video_table=video_table,
    best_day=best_day,
    prediction=predicted_views,
    avg_engagement=avg_engagement,
    total_videos=total_videos,
    total_views=total_views,
    r2=r2,
    mae=mae
)


# -----------------------------------------
# APPLICATION ENTRY POINT
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)