import streamlit as st
import pandas as pd
import numpy as np
import chromadb, os
import chromadb.utils.embedding_functions as embedding_functions
from threading import Thread
from helpers import *
from youtube import get_youtube_video_df

DF_PATH = "files/videos.csv"
VIDEO_EMB_COLLECTION = "videos-0926-large-512"
TERMS_EMB_COLLECTION = "terms-large-512"
EMB_MODEL_NAME = "text-embedding-3-large"
EMB_DIMENSIONS = 512
CHROMA_API_KEY = os.environ['CHROMA_API_KEY']
CHROMA_TENANT = os.environ['CHROMA_TENANT']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
N_TERMS = 3

# Set up to df / youtube
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DF_PATH)
    
    # Load the latest videos on background thread
    Thread(target=get_youtube_video_df, args=(st.session_state, DF_PATH), daemon=True).start()

    # Sort by publication
    st.session_state.df['published_at_datetime'] = pd.to_datetime(st.session_state.df['published_at'], utc=True)
    st.session_state.df = st.session_state.df.sort_values(by='published_at_datetime', ascending=False)
    # add a channel_title: title column for selectbox
    st.session_state.df['channel_title_with_title'] = st.session_state.df['channel_title'].str.cat(st.session_state.df['title'], sep=': ')

    st.session_state.client = chromadb.CloudClient(
      api_key=CHROMA_API_KEY,
      tenant=CHROMA_TENANT,
      database='Youtube'
    )

    st.session_state.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
      api_key=OPENAI_API_KEY,
      model_name=EMB_MODEL_NAME,
      dimensions=EMB_DIMENSIONS
    )
    st.session_state.video_collection = st.session_state.client.create_collection(
      name=VIDEO_EMB_COLLECTION,
      embedding_function=st.session_state.openai_ef, 
      get_or_create=True  
    )
    st.session_state.term_collection = st.session_state.client.create_collection(
        name=TERMS_EMB_COLLECTION, 
        embedding_function=st.session_state.openai_ef, 
        get_or_create=True  
    )


# CSS design tweaks
st.markdown(
    """
    <style>
    h6, h5 {
        padding-bottom: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    p {
        margin-bottom: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stCaptionContainer"] {
        margin-bottom: -12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    button[kind="secondary"], button[kind="primary"], .stButton > button {
        padding-top: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




title = st.selectbox(
    "Search YouTube Video Title",
    st.session_state.df['channel_title_with_title'].to_list(),
    index=0,
    placeholder="Video title",
    accept_new_options=False,
)


filtered_df = st.session_state.df
if title:
  filtered_df = filtered_df.loc[filtered_df['channel_title_with_title'] == title]
  print(len(filtered_df), filtered_df.iloc[:5]['title'])

video = None
if len(filtered_df):
  video = filtered_df.iloc[0]

# Title and Thumbnail
vid_data = []
if video is not None:
  title_row = st.columns([1,3])
  with title_row[0]:
    st.image(video.loc['thumbnail'] , width='stretch')
  
  with title_row[1]:
    title_tile = st.container(border=True, key="title", height=126)  
    print(video)
    title_tile.markdown(f"##### {video.loc['title']}")
    title_tile.caption(f"{video.loc['channel_title']}")


if video is not None:
  # Capture video info
  channel_df = st.session_state.df[st.session_state.df['channel_id'] == video['channel_id']]
  channel_views = channel_df['views'].sum()
  channel_median_views = channel_df['views'].mean()
  channel_vids_count = channel_df['id'].count()

  vid_data = [
    [
      ('Views', format_number(video.loc['views'])), 
      ('Likes', format_number(video.loc['likes'])), 
      ('Comments', format_number(video.loc['comments'])), 
    ],
    [
      ('Channel views', format_number(channel_views)), 
      ('Avg channel views', format_number(channel_median_views)), 
      ('Subscribers', format_number(video.loc['subscriber_count']))
    ]
  ]

  # Dispaly Video info
  for r, row_data in enumerate(vid_data):
    row1 = st.columns(3)
    for i, col in enumerate(row1):
      with col:
        if len(row_data) > i:
          tile = st.container(border=True, height=60)
          data = row_data[i]
          tile.markdown(f"###### {data[1]} {data[0]}")


# Top/bottom n prediction
if video is not None:
  top_n = predict_top_n_percent(
    st.session_state.df, 
    video, 
    st.session_state.video_collection, 
    st.session_state.openai_ef
  )
  if top_n is not None:
    tile = st.container(border=True)
    top_bottom = "Bottom" if top_n <= 0.5 else "Top"
    # Want >= 50%/25% not <=
    if top_n > 0.5:
      top_n = (1-top_n)+0.25
    tile.markdown(f"{top_bottom} {int(top_n*100)}% in channel views")
    tile.caption(f"Video Prediction")


# Title Rewrite
if video is not None:
  new_title = None
  if st.session_state.get('Rewrite'):
    col, = st.columns(1)
    new_title = rewrite_title(video, OPENAI_API_KEY)

  title_rewrite = st.button("Rewrite Title", width='stretch', key='Rewrite')
  if title_rewrite:
    if new_title:
      tile = col.container(border=True)
      tile.markdown(new_title)
      tile.caption(f"Title Rewrite")
    


# Features
if video is not None:
  # button = st.button("See Impactful Features", width='stretch')

  space = st.container(border=False, height=12)

  # New feature
  new_feature = st.text_input("Feature impact", placeholder='Test your own feature (e.g. Viral title, Exciting product, Popular guest)')

  row2 = st.columns(3)
  # if button:

  # Get video embedding
  video_emb_docs = get_from_chroma_with_ids(
    st.session_state.video_collection, 
    [video['id']]
  )

  # Upload embedding if it doesn't exist
  if not len(video_emb_docs['embeddings']):
    print(f"Uploading embedding")

    add_emb_to_chroma(
      st.session_state.video_collection, 
      video['id'], 
      None, 
      video['embedding_text']
    )
    video_emb_docs = get_from_chroma_with_ids(
      st.session_state.video_collection, 
      [video['id']]
    )

  # Query with video embedding
  top_term_docs = st.session_state.term_collection.query(
    query_embeddings=video_emb_docs['embeddings'],
    n_results=N_TERMS,
  )
  print(top_term_docs)

  if top_term_docs.get('documents'):
    top_terms = top_term_docs['documents'][0]
    dist = top_term_docs.get('distances')
    dist = dist[0] if dist else dist

    for i, col in enumerate(row2):
      with col:
        if len(top_terms) > i and format_impact(dist[i]):
          tile = st.container(border=True)
          term = top_terms[i].replace('episode', '').replace('Episode', '')
          tile.markdown(f"###### {term}")

          if len(dist) > i:
            tile.caption(f"Impact: {format_impact(dist[i])}")

  
  if new_feature:
    try:

      # Get video embedding
      video_emb_docs = get_from_chroma_with_ids(
        st.session_state.video_collection, 
        [video['id']]
      )
      video_emb = video_emb_docs.get('embeddings')[0]

      # Get feature embedding
      feature_emb = np.array(st.session_state.openai_ef(new_feature)[0])
      # Compute squared L2 distance between embeddings (default Chroma distance)
      term_dist = np.sum((video_emb - feature_emb) ** 2)

      print(term_dist)

      if term_dist:
        # row3 = st.columns(1)
        # with row3:
        tile = st.container(border=True)
        tile.markdown(f"###### {new_feature}")
        tile.caption(f"Impact: {format_impact(term_dist)}")


    except Exception as e:
      st.error(f"Error generating embedding: {e}")


if video is not None:
  space = st.container(border=False, height=20)
  container = st.container()
  plot_channel_over_time(
    container, st.session_state.df, video['channel_id'], video['id']
  )
  plot_channel_duration_over_time(
    container, st.session_state.df, video['channel_id']
  )
  plot_work_per_video_type(
    container, st.session_state.df, video['channel_id']
  )

# Description
if video is not None:
  space = st.container(border=False, height=4)

  title_tile = st.container(border=True)  
  title_tile.markdown(f"##### Description \n{video.loc['description']}")