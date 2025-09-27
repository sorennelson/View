import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from helpers import format_number, format_impact, get_from_chroma_with_ids

CHROMA_API_KEY = 'ck-CL2eRYFeJAN8mTfL1Xup2Bun4nrKgDht4RAzDqia7uT'
CHROMA_TENANT = '0b1b6a8d-f2d5-4dd8-b9ce-5aae12272521'
VIDEO_EMB_COLLECTION = "videos-0926-large-512"
TERMS_EMB_COLLECTION = "terms-large-512"
OPENAI_API_KEY = "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"
N_TERMS = 3

# Set up to df / youtube
if "video_df" not in st.session_state:
    st.session_state.df = pd.read_csv("videos.csv")

    st.session_state.client = chromadb.CloudClient(
      api_key=CHROMA_API_KEY,
      tenant=CHROMA_TENANT,
      database='Youtube'
    )

    st.session_state.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
      api_key=OPENAI_API_KEY,
      model_name="text-embedding-3-large",
      dimensions=512
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
    



st.markdown(
    """
    <style>
    h6 {
        padding-bottom: 8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     """
#     <style>
#     .st-key-title  {
#         margin-top: 15px !important;
#         margin-left: 15px !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )



# Title search
# title = st.text_input("Search YouTube Video Title")
# filtered_df = st.session_state.df
# if title:
#   filtered_df = filtered_df.loc[filtered_df["title"].str.contains(title, case=False, na=False)]

title = st.selectbox(
    "Search YouTube Video Title",
    st.session_state.df['title'].to_list(),
    index=0,
    placeholder="Video title",
    accept_new_options=False,
)


filtered_df = st.session_state.df
if title:
  filtered_df = filtered_df.loc[filtered_df["title"] == title]
  print(len(filtered_df), filtered_df.iloc[:5]['title'])

video = None
if len(filtered_df):
  video = filtered_df.iloc[0]

vid_data = []
# Title
if video is not None:
  title_row = st.columns([1,3])
  with title_row[0]:
    st.image(video.loc['thumbnail'] , width='stretch')
  
  with title_row[1]:
    title_tile = st.container(border=True, key="title")  
    print(video)
    title_tile.markdown(f"##### {video.loc['title']}")
    title_tile.caption(f"{video.loc['channel_title']}")

    channel_df = st.session_state.df[st.session_state.df['channel_id'] == video['channel_id']]
    channel_views = channel_df['views'].sum()
    channel_mean_views = channel_df['views'].mean()
    channel_vids_count = channel_df['id'].count()

    vid_data = [
      [
        ('Views', format_number(video.loc['views'])), 
        ('Likes', format_number(video.loc['likes'])), 
        ('Comments', format_number(video.loc['comments'])), 
      ],
      [
        ('Channel views', format_number(channel_views)), 
        ('Avg channel views', format_number(channel_mean_views)), 
        # ('Channel Videos', format_number(channel_vids_count)), 
        ('Subscribers', format_number(video.loc['subscriber_count']))
      ]
    ]

# Video info
for r, row_data in enumerate(vid_data):
  row1 = st.columns(3)
  for i, col in enumerate(row1):
    with col:
      if len(row_data) > i:
        tile = st.container(border=True, height=60)
        data = row_data[i]
        tile.markdown(f"###### {data[1]} {data[0]}")
      # else:
      #   tile.markdown(f"###### --")


# Features
if video is not None:
  # button = st.button("See Impactful Features", width='stretch')

  space = st.container(border=False, height=12)

  # New feature
  new_feature = st.text_input("Feature impact", placeholder='e.g. Viral title')

  row2 = st.columns(3)
  # if button:

  # Get video embedding
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
        if len(top_terms) > i:
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
        row3, _ = st.columns([1,1])
        with row3:
          tile = st.container(border=True)
          tile.markdown(f"###### {new_feature}")
          tile.caption(f"Impact: {format_impact(term_dist)}")


    except Exception as e:
      st.error(f"Error generating embedding: {e}")


  

  
  row4 = st.columns(4)


# Description
if video is not None:
  space = st.container(border=False, height=12)

  title_tile = st.container(border=True)  
  title_tile.markdown(f"##### Description \n{video.loc['description']}")