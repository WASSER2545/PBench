

import numpy as np
import pandas as pd


def create_pool():
    all_db_name = get_all_db_name()
    pool=pd.DataFrame(
        columns=["database","sql","avg_cpu_time","avg_scan_bytes","avg_duration","filter","join","agg","sort"]
    )
    #这里借用一下ilp里面的函数
    #记得加入output/additional_query.csv
    return pool


def distance(virtual_query, sql):
    """ Calculate the distance between the virtual query and the SQL. """
    weight=[1,1,1,1,1,1,1,1]
    distance+=abs(virtual_query["avg_cpu_time"]-sql["avg_cpu_time"])*weight[0]
    distance+=abs(virtual_query["avg_scan_bytes"]-sql["avg_scan_bytes"])*weight[1]
    distance+=abs(virtual_query["avg_duration"]-sql["avg_duration"])*weight[2]
    distance+=abs(virtual_query["filter"]-sql["filter"])*weight[3]
    distance+=abs(virtual_query["join"]-sql["join"])*weight[4]
    distance+=abs(virtual_query["agg"]-sql["agg"])*weight[5]
    distance+=abs(virtual_query["sort"]-sql["sort"])*weight[6]
    return distance


#这个应该不会做到……谁没事又搞一个sbert
def sbert_distance(virtual_query, sql):
    """ Calculate the distance between the virtual query and the SQL. """
    return distance

def find_k_nearest_neighbors(pool, virtual_query, k):
    """ Find the k-nearest neighbors in the pool. """
    #计算query和pool里面的每一个sql的距离，距离函数很重要
    return k_nearest_neighbors

def find_k_distants_neighbors(pool, virtual_query, k):
    """ Find the k-distants neighbors in the pool. """
    #计算query和pool里面的每一个sql的距离，距离函数很重要
    return k_distants_neighbors

