#!/usr/bin/python
import re

import psycopg2.extras
import logging
from pymongo import MongoClient, UpdateOne
from env import config
import psycopg2 as pg
import pandas as pd
from bson.objectid import ObjectId

def connect():
    params = config.config()
    connection = pg.connect(**params, cursor_factory=pg.extras.DictCursor)
    return connection


def execute(sql, params={}):
    with connect() as connection:
        with connection.cursor(cursor_factory=pg.extras.DictCursor) as cursor:
            cursor.execute(sql, params)


def mongo_client_connect():
    params = config.mongo_config()
    mongoClient = MongoClient(host=params['host'], port=int(params['port']), unicode_decode_error_handler='ignore')
    return mongoClient

def mongo_db_connect():
    params = config.mongo_config()
    client = mongo_client_connect()
    db = client[params['database']]
    return db

def mongo_collection_connect():
    params = config.mongo_config()
    db = mongo_db_connect()
    collection = db[params['collection']]
    return collection

def bulk_token_update(token_df):
    update_query_list = []
    collection = mongo_collection_connect()
    for n in range(len(token_df)):
        query = {"_id": ObjectId(token_df['_id'][n])}
        new_data = {
            "$set":
                {
                    "sentiment": {
                        "positive": {
                            "score": token_df['p_score'][n],
                            "words": token_df['p_words'][n]
                        },
                        "negative": {
                            "score": token_df['n_score'][n],
                            "words": token_df['n_words'][n]
                        }
                    },
                    "sentimented": int(token_df['sentimented'][n])
                }
            }
        update_query_list.append(UpdateOne(query, new_data))

    collection.bulk_write(update_query_list)


def filtered_find():
    results = mongo_collection_connect().find(
        {
            "filtered": 1,
        },
        {
            "_id": 1,
            "point": 1,
            "content": 1
        }
    )
    return results

# def save_emotion():
