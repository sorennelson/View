import pandas as pd
import altair as alt
import numpy as np
import pickle, sklearn, openai, tiktoken

MAX_TOKENS = 8000  # safe under the 8191 limit
MODEL_PATH = "files/model-uplift-0927.pkl"
EMB_MODEL_NAME = "text-embedding-3-large"


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
  elif distance < 1.3:
    return 'Medium'
  elif distance < 1.4:
    return 'Low'
  else: 
    return None

def truncate_text(text, max_tokens=MAX_TOKENS):
    tokenizer = tiktoken.encoding_for_model(EMB_MODEL_NAME)
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

def get_from_chroma_with_ids(collection, ids):
    data = collection.get(ids=ids, include=["documents", "embeddings"])
    return data

def add_emb_to_chroma(collection, id, emb, doc):
    collection.add(
        documents=[truncate_text(doc)],
        ids=[id],
        embeddings=[emb] if emb is not None else None
    )

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
    channel_videos['published_at_datetime'] = pd.to_datetime(channel_videos['published_at'], utc=True)
    channel_median_views = channel_videos[channel_videos['published_at_datetime'] > time_cutoff]['views'].median()

  # Load video embedding from chroma
  video_docs = get_from_chroma_with_ids(video_collection, [video['id']])

  # If video isn't in chroma then create embedding
  if not video_docs.get('documents'):
    emb = embedder(truncate_text(video['embedding_text']))
    emb = emb[0]
    # Store in Chroma
    print("Adding to chroma")
    add_emb_to_chroma(video_collection, video['id'], emb, video['embedding_text'])
    
  else:
    print("Embedding in chroma")
    emb = video_docs['embeddings'][0]

  # Predict
  pred = regr.predict([emb])
  if type(pred) in [np.ndarray, list] and len(pred):
    pred = pred[0]

  print(f"Predicted uplift: {np.exp(pred)}")
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


def rewrite_title(video, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)

    prompt = f"Using the following YouTube title and description, rewrite the title to boost views. Keep the title professional as if you were creating this for @{video['channel_title']}. Remember, the goal is to boost views so tailor yourself to the channel's audience. Do not add information you cannot get from the information provided. Respond with only the title and no other dialog. \n\n {video['embedding_text']}"

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    print(prompt)
    if type(response) == str:
        return response
    if not hasattr(response, "output"):
        return None
    output = response.output
    if not output and len(output) < 2:
         return None
    output = output[1].content
    if not output:
         return None
    return output[0].text


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
        'Views': channel_videos.values,
        "Cumulative Views": channel_videos.cumsum()
    })

    # Plot
    pdf = alt.Chart(df).mark_area(opacity=0.5).encode(
        x=alt.X(
            'Month:T', 
            axis=alt.Axis(format='%b %Y', tickCount=6, title="Publication Date")
        ),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Channel Views")
        ),
        tooltip=[
            alt.Tooltip('Month:T', title='Month', format='%b %Y'),
            alt.Tooltip('Views', title='Views', format=',.0f'),
        ]
    )

    # line at top of area
    pdf_line = alt.Chart(df).mark_line(
        opacity=0.8,
        strokeWidth=2
    ).encode(
        x='Month:T',
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Channel Views")
        )
    )

    cdf_time = (
        alt.Chart(df)
        .mark_area(color="steelblue", opacity=0.3)
        .encode(
            x="Month:T",
            y=alt.Y('Cumulative Views:Q',
                axis=alt.Axis(tickCount=6, title="Channel Cumulative Views")
            ),
        )
    )
    cdf_time_line = (
        alt.Chart(df)
        .mark_line(color="steelblue", opacity=0.8)
        .encode(
            x="Month:T",
            y=alt.Y('Cumulative Views:Q',
                axis=alt.Axis(tickCount=6, title="Channel Cumulative Views")
            ),
        )
    )

    chart = alt.layer(pdf + pdf_line, cdf_time + cdf_time_line
    ).resolve_scale(
      y='independent'
    ).properties(
        title="Is my channel growing? (Monthly channel views by publication)"
    ).configure_title(
        fontSize=16,
        anchor='middle',
    )
    container.altair_chart(chart.interactive(bind_y=False), use_container_width=True)
    # _, col, _ = container.columns(3)
    # col.caption("Channel Views By Publication")


def plot_channel_duration_over_time(container, df_videos, channel_id):
    ''' Plot the duration of videos over time for a channel '''

    channel_videos = df_videos[df_videos['channel_id'] == channel_id]
    n_videos_lt_5 = len(channel_videos[channel_videos['duration'] < 5*60])
    n_videos_gt_5_lt_20 = len(channel_videos[(channel_videos['duration'] >= 5*60) & (channel_videos['duration'] < 20*60)])
    n_videos_gt_20 = len(channel_videos[channel_videos['duration'] >= 20*60])
    print(f"Total videos: {len(channel_videos)}")
    print(f"Short form - Number of videos < 5 minutes {n_videos_lt_5}")
    print(f"Medium - Number of videos 5 < x <= 20 minutes {n_videos_gt_5_lt_20}")
    print(f"Long form - Number of videos >= 20 minutes {n_videos_gt_20}")

    channel_videos = channel_videos.sort_values(by='duration').reset_index()

    # convert to a DataFrame
    df = pd.DataFrame({
        'Video': channel_videos['title'], 
        'Views': channel_videos['views'],
        "Duration": channel_videos['duration'] / 60,
        "Order": range(len(channel_videos)),
    })

    # Plot
    video_views_area_chart = alt.Chart(df).mark_area(opacity=0.5).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Video Views")
        ),
        tooltip=[
            alt.Tooltip('Video', title='Video'),
            alt.Tooltip('Views', title='Views', format=',.0f'),
            alt.Tooltip('Duration', title='Duration (min)', format=',.0f')
        ]
    )

    # line at top of area
    video_views_line_chart = alt.Chart(df).mark_line(
        opacity=0.8,
        strokeWidth=2
    ).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Video Views")
        )
    )

    video_duration_area_chart = alt.Chart(df).mark_area(
        color="steelblue", 
        opacity=0.3
    ).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Duration:Q',
            axis=alt.Axis(tickCount=6, title="Duration (minutes)")
        )
    )
    video_duration_line_chart = alt.Chart(df).mark_line(
        color="steelblue", 
        opacity=0.8
    ).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Duration:Q',
            axis=alt.Axis(tickCount=6, title="Duration (minutes)")
        ),
    )
    
    # display
    chart = alt.layer(
        video_views_area_chart + video_views_line_chart,
        video_duration_area_chart + video_duration_line_chart,
        data=df
    ).resolve_scale(
        y='independent'
    ).interactive(
      bind_y=False
    ).properties(
        title="How many viewers are watching videos this long? (Video views by duration)"
    ).configure_title(
        fontSize=16,
        anchor='middle',
    )
    container.altair_chart(chart, use_container_width=True)


def plot_work_per_video_type(container, df_videos, channel_id):
    
    channel_videos = df_videos[df_videos['channel_id'] == channel_id]
    channel_videos['video_type'] = channel_videos['duration'].apply(
        lambda x: 'Short (<3M)' if x < 3*60 else 'Clip (<20M)' if x < 20*60 else 'Long Form (≥20M)'
    )
    video_type_groups = channel_videos.groupby('video_type')
    # Get per-group median
    video_type_views = video_type_groups['views'].median()
    work_per_video = video_type_groups['duration'].median() / 60 * 5

    df = pd.DataFrame({
        'Video Type': video_type_views.index, 
        'Views': video_type_views.values,
        'Work': work_per_video.values,
    })
    # Define the desired order for video types
    video_type_order = ['Short (<3M)', 'Clip (<20M)', 'Long Form (≥20M)']
    # Altair chart that shows a scatter plot where each column is a video_type_view, size is a the total duration * 5, and height is video_type_view
    scatter = alt.Chart(df).mark_circle().encode(
        x=alt.X('Video Type:N', sort=video_type_order,
                axis=alt.Axis(labels=True, ticks=False, labelAngle=0)
        ),
        y=alt.Y(
            'Work:Q',
            scale=alt.Scale(domain=[-100, df['Work'].max() + 200]),
            axis=alt.Axis(tickCount=6, title="Production Work (5 Min/Median Duration Min)")
        ),
        size=alt.Size(
            'Views:Q',
            scale=alt.Scale(range=[1000, 10000]),
            # legend=alt.Legend(title="Production work (minutes*5)")
            legend=None
        ),
        color=alt.Color(
            'Video Type:N',
            # scale=alt.Scale(domain=work_per_video.values),  # only these three show in the legend
            legend=None
        ),
        tooltip=[
            alt.Tooltip('Video Type:N', title='Type'),
            alt.Tooltip('Views:Q', title='Median Views', format=',.0f'),
            alt.Tooltip('Work:Q', title='Estimated Work (min)', format=',.0f')
        ]
    )

    labels = alt.Chart(df).mark_text(
        align='center', 
        baseline='middle',  
        color='white'
    ).encode(
        x=alt.X('Video Type:N', sort=video_type_order),
        y='Work:Q',
        text=alt.Text('Views:Q', format=',.0f'),
        tooltip=[
            alt.Tooltip('Video Type:N', title='Type'),
            alt.Tooltip('Views:Q', title='Median Views', format=',.0f'),
            alt.Tooltip('Work:Q', title='Estimated Work (min)', format=',.0f')
        ]
    )

    # Combine scatter and labels
    chart = alt.layer(
      scatter + labels, 
      data=df
    ).properties(
        title="Is the effort worth the views? (Median views per production cost)",
        height=400
    ).configure_title(
        fontSize=16,
        anchor='middle',
    )

    container.altair_chart(chart, use_container_width=True)