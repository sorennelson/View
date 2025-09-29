import pandas as pd
import altair as alt
import pickle, sklearn
import numpy as np

MAX_TOKENS = 8000  # safe under the 8191 limit
MODEL_PATH = "files/model-uplift-0927.pkl"


def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1000:
        return f"{n/1000:.0f}K"
    else:
        return str(n)

def format_impact(distance):
  if distance < 1.2:
    return 'High'
  elif distance < 1.4:
    return 'Medium'
  return 'Low'

def truncate_text(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

def get_from_chroma_with_ids(collection, ids):
    data = collection.get(ids=ids, include=["documents", "embeddings"])
    return data

def predict_top_n_percent(df_videos, video, video_collection, embedder, n_days_ago=90, regr=None):
  ''' Returns whether the video lies in the top n percent of channel videos. 
  The video must be published within the last n days and be in the bottom 50% of views.
  TODO: Verify if this works on videos not in the csv
  '''
  if regr is None:
    with open(MODEL_PATH, 'rb') as f:
      regr = pickle.load(f)

    if regr is None:
        return None

  # Check if more recent than n days ago
  video = video.copy()
  video['published_at_datetime'] = pd.to_datetime(video['published_at'], utc=True)
  time_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=n_days_ago)
  if video['published_at_datetime'] < time_cutoff:
    return None

  # Sort channel by views
  channel_videos = df_videos[df_videos['channel_id'] == video['channel_id']]
  channel_videos_sorted = channel_videos.sort_values(by='views', ascending=False)
  
  if 'split' in channel_videos.columns:
    # If we trained on channel, use the train median
    channel_median_views = channel_videos[channel_videos['split'] == 'train']['views'].median()
  else:
    # Otherwise grab median of views > time_cutoff
    channel_videos['published_at_datetime'] = pd.to_datetime(video['published_at'], utc=True)
    channel_median_views = channel_videos[channel_videos['published_at_datetime'] > time_cutoff]['views'].median()

  # Load video embedding from chroma
  video_docs = get_from_chroma_with_ids(video_collection, [video['id']])

  # If video isn't in chroma then create embedding
  if not video_docs.get('documents'):
    emb = embedder(truncate_text(video['embedding_text']))
  else:
    emb = video_docs['embeddings'][0]

  # Predict
  pred = regr.predict([emb])
  if type(pred) in [np.ndarray, list] and len(pred):
    pred = pred[0]

  print(f"Predicted uplift: {np.expm1(pred)}")
  # Denormalize into view space
  if pred == 0:
    pred += 0.00001
  pred = np.exp(pred) * channel_median_views
  print(f"Predicted views: {pred}")

#   pred = min(int(pred), video['views'])

  print(f"Max views for channel: {channel_videos_sorted.iloc[0]['views']}")
  print(f"Min views for channel: {channel_videos_sorted.iloc[-1]['views']}")
  print(f"Median views for channel: {channel_median_views}")

  # "bottom 25% of views" "bottom 50% of views" "top 50% of views" "top 25% of views"
  for n in [0.75, 0.5, 0.25, 0.]:
    top_n_percent_idx = int(len(channel_videos_sorted)*n)
    top_n_percent_views = channel_videos_sorted.iloc[top_n_percent_idx]['views']
    # print(f"Top n-5% views for channel: {top_n_percent_views}")
    if pred < top_n_percent_views:
        return 1-n
  
  return None


def plot_channel_over_time(container, df_videos, channel_id, video_id):
    channel_videos = df_videos[df_videos['channel_id'] == channel_id]
    channel_name = channel_videos.iloc[0]['channel_title']
    print(f"Channel name: {channel_name}")
    print(f"Channel ID: {channel_id}")
    print(f"Total videos: {len(channel_videos)}")

    channel_videos['published_at_datetime'] = pd.to_datetime(channel_videos['published_at'], utc=True)
    channel_videos = channel_videos.sort_values(by='published_at_datetime')
    # convert datetime to month and year
    channel_videos['published_at_month'] = channel_videos['published_at_datetime'].dt.to_period('M')

    # Get video published_at_month
    video_month = channel_videos[channel_videos['id'] == video_id].iloc[0]['published_at_month']
    print(f"Video published at month: {video_month}")

    # group by month and compute sum of views
    channel_videos = channel_videos.groupby('published_at_month')['views'].sum()
    channel_videos.index = channel_videos.index.astype(str)
    # channel_videos = channel_videos / 1000
    # convert to a DataFrame
    df = pd.DataFrame({
        'Month': channel_videos.index, 
        'Views': channel_videos.values
    })

    # Plot
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(
            'Month:T', 
            axis=alt.Axis(format='%b %Y', tickCount=6, title="Publication Date")
        ),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Channel Views")
        )
    ).interactive()

    container.altair_chart(chart, use_container_width=True)
    # _, col, _ = container.columns(3)
    # col.caption("Channel Views By Publication")
