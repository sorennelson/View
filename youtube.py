import isodate, os, datetime
import pandas as pd
from pydantic import BaseModel
from typing import List
from googleapiclient.discovery import build


youtube = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])

creator_names = [
  "CalNewportMedia", "@DailyStoic", "timferriss", "hubermanlab", "lexfridman", 
  "OpenAI", "hardfork", "MastersofScale_", "TheRobotBrainsPodcast", "googledeepmind", "anthropic-ai"
]


class Video(BaseModel):
    title: str
    description: str
    tags: list[str]
    thumbnail: str
    kind: str
    id: str
    channel_id: str
    channel_title: str
    subscriber_count: int
    category_id: int
    duration: int
    is_captioned: bool
    published_at: str
    views: int
    likes: int
    comments: int
    license: str
    embedding_text: str


def get_youtube_video_df(session_state, path):
    if os.path.exists(path):
        # check if modified in past day
        modified_ts = os.path.getmtime(path)
        modified_time = datetime.datetime.fromtimestamp(modified_ts)
        print("Last modified:", modified_time)
        cutoff = datetime.datetime.now() - datetime.timedelta(minutes=5)
        do_get_stats = modified_time <= cutoff
    else:
        do_get_stats = True

    if not do_get_stats:
        return

    try:
        channel_stats = []
        for creator_name in creator_names:
            channel_name, channel_id = get_channel_id(creator_name)
            print(f"Channel name: {channel_name}, ID: {channel_id}")
            print(f"... getting videos")
            channel_stats.extend(get_stats(channel_id))

        df = pd.DataFrame([v.model_dump() for v in channel_stats])
        df = df.drop_duplicates(subset=['id'])
        df.to_csv(path, index=False)
        session_state.df = df
    except Exception as e:
        print(f"Failed to update videos DF")



def get_channel_id(creator_name: str):
    resp = youtube.search().list(
        part="snippet",
        q=creator_name,
        type="channel"
    ).execute()

    if not resp["items"]:
        raise ValueError(f"No channel found with name {creator_name}")

    return (resp["items"][0]['snippet']['title'], resp["items"][0]["id"]["channelId"])


def parse_time(timestr: str) -> int:
    """Return total seconds from ISO 8601 duration string like PT2H18M30S."""
    return int(isodate.parse_duration(timestr).total_seconds())


def get_embedding_text(title: str, creator: str, duration: str, published_at: str, description: str) -> str:
    return f"# YouTube Video \n\n## Channel Name: {creator} \n\n## Title: {title}\n\n## Duration: {duration}\n\n## Published on: {published_at}\n\n## Description:\n{description}"


def get_stats(channel_id: str):
    # Get uploads playlist ID
    channel_resp = youtube.channels().list(
        part="contentDetails,statistics",
        id=channel_id
    ).execute()
    uploads_playlist_id = channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    print(f"Statistics {channel_resp['items'][0]['statistics']}")
    subscriber_count = channel_resp['items'][0]['statistics']['subscriberCount']
    subscriber_count = int(subscriber_count) if subscriber_count else 0

    # Collect video IDs from uploads playlist
    video_ids = []
    next_page = None
    while True:
        playlist_resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page
        ).execute()

        for item in playlist_resp["items"]:
            video_ids.append(item["contentDetails"]["videoId"])

        next_page = playlist_resp.get("nextPageToken")
        if not next_page:
            break

    video_views = get_video_stats(video_ids, subscriber_count)
    return video_views


def get_video_stats(video_ids: List[str], subscriber_count: int):
    # Get stats for video IDs (50 per call)
    video_views = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        videos_resp = youtube.videos().list(
            part="statistics,snippet,contentDetails,status",
            id=",".join(batch)
        ).execute()

        for item in videos_resp["items"]:
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            duration = item["contentDetails"]["duration"]
            channel_title = item["snippet"]["channelTitle"]
            published_at = item["snippet"]["publishedAt"]

            thumbnails = item["snippet"]["thumbnails"].get('high', None)
            if not thumbnails:
                thumbnails = item["snippet"]["thumbnails"].get('medium', None)
            if not thumbnails:
                thumbnails = item["snippet"]["thumbnails"].get('default', None)
            thumbnail = thumbnails.get('url', None) if thumbnails else None

            if not description:
                # Ignore any videos without descriptinos
                continue

            vid = Video(
                title=title,
                description=description,
                tags=item["snippet"].get("tags", []),
                thumbnail=thumbnail,
                kind=item["kind"],
                id=item["id"],
                channel_id=item["snippet"]["channelId"],
                channel_title=channel_title,
                subscriber_count=subscriber_count,
                duration=parse_time(duration),
                category_id=item["snippet"]["categoryId"],
                is_captioned=item["contentDetails"]["caption"],
                published_at=published_at,
                views=int(item["statistics"].get("viewCount",0)),
                likes=int(item["statistics"].get("likeCount", 0)),
                comments=int(item["statistics"].get("commentCount", 0)),
                license=item["status"]["license"],
                embedding_text=get_embedding_text(title, channel_title, duration, published_at, description)
            )

            video_views.append(vid)

    print(f"Fetched {len(video_views)} videos")

    return video_views