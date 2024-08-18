import streamlit as st
from sqlalchemy import func
from pandas import DataFrame, Timedelta
from util.yt_sql import (
    sql, 
    YtChannelModel as ch, 
    YtVideosModel as v, 
    YtCommentsModel as cm
)
from util.yt_df import df_with_link


session = sql()


st.markdown('## Get answers for questions:')


with st.expander("1 - What are the names of all the videos and their corresponding channels?"):
    
    result = session.query(
        v.title.label('Video Name'), ch.channel_name.label('Channel Name'),
        func.concat('https://www.youtube.com/watch?v=', v.video_id).label('Link')
    ).join(ch, v.channel_id == ch.channel_id) 
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)
    df_with_link(df, 'Video Name', 'Link')


with st.expander("2 - Which channels have the most number of videos, and how many videos do they have?"):
    
    result = session.query(
        ch.channel_name, ch.thumbnail, ch.vid_count
    ).order_by(ch.vid_count.desc()).first()

    st.write(result[0])
    # st.image(result[1], )
    st.write(f'{result[0]} channel has the most number of videos that is {result[2]}')


with st.expander("3 - What are the top 10 most viewed videos and their respective channels?"):
    
    result = session.query(
        v.title.label('Video Name'), v.view_count.label('Views'), ch.channel_name.label('Channel Name'),
        func.concat('https://www.youtube.com/watch?v=', v.video_id).label('Link')
    ).join(ch, v.channel_id == ch.channel_id).order_by(v.view_count.desc()).limit(10)
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)
    df_with_link(df, 'Video Name', 'Link')


with st.expander("4 - How many comments were made on each video, and what are their corresponding video names?"):
    
    result = session.query(
        v.title.label('Video Name'), v.comment_count.label('Comments'),
        func.concat('https://www.youtube.com/watch?v=', v.video_id).label('Link')
    )
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)
    df_with_link(df, 'Video Name', 'Link')


with st.expander("5 - Which videos have the highest number of likes, and what are their corresponding channel names?"):
    
    result = session.query(
        v.title.label('Video Name'), v.like_count.label('Likes'), ch.channel_name.label('Channel Name'),
        func.concat('https://www.youtube.com/watch?v=', v.video_id).label('Link')
    ).join(ch, v.channel_id == ch.channel_id).order_by(v.like_count.desc()).limit(10)
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)
    df_with_link(df, 'Video Name', 'Link')


with st.expander("6 - What is the total number of likes and dislikes for each video, and what are their corresponding video names?"):
    
    result = session.query(
        v.title.label('Video Name'), v.like_count.label('Likes'),
        func.concat('https://www.youtube.com/watch?v=', v.video_id).label('Link')
    )
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)
    df_with_link(df, 'Video Name', 'Link')


with st.expander("7 - What is the total number of views for each channel, and what are their corresponding channel names?"):
    
    result = session.query(ch.channel_name, ch.view_count)
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)
    st.write(df)


with st.expander("8 - What are the names of all the channels that have published videos in last month?"):
    
    result = session.query(
        ch.channel_name.label('Channel'), v.published.label('Published'), 
    ).join(v, v.channel_id == ch.channel_id)
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)
    df = df[df['Published'].dt.month == 2].groupby('Channel').count()
    df = df.rename(columns={'Published': 'Count of Videos'})
    st.write(df)


with st.expander("9 - What is the average duration of all videos in each channel, and what are their corresponding channel names?"):
    
    result = session.query(
        ch.channel_name.label('Channel'), 
        func.concat('PT', v.duration).label('Duration'),
    ).join(v, v.channel_id == ch.channel_id)
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)

    df['Duration'] = df['Duration'].apply(Timedelta).dt.total_seconds()
    df = df.groupby('Channel').mean()
    df = df.rename(columns={'Duration': 'Average video duration (s)'})

    st.write(df)


with st.expander("10 - Which videos have the highest number of comments, and what are their corresponding channel names?"):
    
    result = session.query(
        v.title.label('Video Name'), 
        v.comment_count.label('Comments'), 
        ch.channel_name.label('Channel'), 
        func.concat('https://www.youtube.com/watch?v=', v.video_id).label('Link')
    ).join(v, v.channel_id == ch.channel_id).order_by(v.comment_count.desc()).limit(10)
    
    df = DataFrame([i for i in result])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1)

    df_with_link(df, 'Video Name', 'Link')

