from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Date, Float,
    )
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
username = os.environ.get('sql_db_usr') 
password = os.environ.get('sql_db_pwd')
hostname = 'localhost'
port = os.environ.get('sql_db_endpoint') 
database = os.environ.get('sql_db_dbname')

# db_channels_coll_name = os.environ.get('db_channels_coll_name')
# db_videos_coll_name = os.environ.get('db_videos_coll_name')
# db_comments_coll_name = os.environ.get('db_comments_coll_name')

# postgre
db_url = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}'
# mysqlclient
# db_url = f'mysql+mysqldb://{username}:{password}@{hostname}/{database}'



# Create engine
engine = create_engine(db_url)


# Create a base class for your ORM models
Base = declarative_base()

# # Function to execute and display the result of a SQL query
# def execute_query(session, query, params):
#     try:
#         cursor.execute(query, params)
#         results = cursor.fetchall()
#         df = pd.DataFrame(results, columns=["Bus_name", "Bus_type", "Start_time", "End_time", "Total_duration",
#                                             "Price", "Seats_Available", "Ratings", "Route_name", "Origin", "Destination"])
#         return df
#     except Exception as e:
#         st.error(f"Error fetching data: {e}")
#         return pd.DataFrame()




# Define your ORM model
class States(Base):
    __tablename__ = 'states'

    state = Column(String, primary_key=True)
    state_link = Column(String)


class Routes(Base):
    __tablename__ = 'routes'

    route_name = Column(String, primary_key=True)
    state = Column(String)
    route_link = Column(String)


class Buses(Base):
    __tablename__ = 'buses'

    bus_id = Column(String, primary_key=True)
    bus_name = Column(String)
    bus_type = Column(String)
    dur = Column(String)
    date_dep = Column(Date)
    date_arrival = Column(Date)
    time_dep = Column(DateTime)
    time_arrival = Column(DateTime)
    star_rate = Column(Float)
    price = Column(Integer)
    seat_available = Column(Integer)
    route_name = Column(String)
    loc_dep = Column(String)
    loc_arrival = Column(String)
    operator = Column(String)




def sql():

    # Create the tables in the database
    Base.metadata.create_all(engine)

    # Create a session maker
    Session = sessionmaker(bind=engine)
    session = Session()
    return session, engine

if __name__ == '__main__':
    pass