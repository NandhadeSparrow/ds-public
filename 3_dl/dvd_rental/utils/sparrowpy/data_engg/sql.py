from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Date, Float,
    MetaData, BigInteger, text
    )
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
username = os.environ.get('sql_db_usr') 
password = os.environ.get('sql_db_pwd')
hostname = 'localhost'
port = os.environ.get('sql_db_endpoint') 
database = os.environ.get('sql_db_dbname')


# postgresql
db_url = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}'


# Create engine
engine = create_engine(db_url)


# Create a base class for your ORM models
Base = declarative_base()


# Define ORM model
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
    print(db_url)
    
    # Create the tables in the database
    # Base.metadata.create_all(engine)

    # Create a session maker
    # Session = sessionmaker(bind=engine)
    # session = Session()
    return engine



def get_table_df(query=None, t_main=None, calc_cols=None, t_joins=None, t_cols=None, groups=None, orders=None):
    engine = sql()

    
    if query is None:
        query = f"select"

        if t_cols:
            cols = []
            for t in t_cols:
                for c in t[1]:
                    cols.append(f"{t[0]}.{c}")
            query += ' ' + ', '.join(cols)
        else:
            query += " *"


        if calc_cols:
            query += ', '+calc_cols


        query += f" from {t_main}"


        if t_joins:
            for t in t_joins:
                query += f" left outer join {t[0]} on {t[1]}"
                # query += f" left outer join {t[0]} on {t_main}.{t[1]} = {t[0]}."
                # if len(t) > 2:
                #     query += t[2]
                # else:
                #     query += t[1]
            

        if groups:
            query += f' group by {', '.join(groups)}'
        if orders:
            query += f' order by {', '.join(orders)}'

    
    print(query)
    stmt = text(query)
    with engine.connect() as connection: res = connection.execute(stmt)
    df = pd.DataFrame(res.fetchall(), columns=res.keys())
    return df

if __name__ == '__main__':
    pass