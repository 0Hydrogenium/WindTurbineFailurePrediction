import pandas as pd
from tqdm import tqdm
import warnings
import logging
import os
import json

from preprocessing.SqlRelated.SqlMethods import SqlMethods

warnings.filterwarnings("ignore")

logger = logging.getLogger()

sql_methods = SqlMethods()

"""
    f: failure
    p: wind_turbine 
"""
f_name = "afval_history"
f_desc_name = "afval"
# 获取所有风机SCADA数据库中的表名称
rval_history_table_names = [x for x in sql_methods.get_all_table_name() if "rval_history" in x]
p_desc_name = "rval"

p_df_path = "preprocessing/temporal/data/{}_df.csv"


class ColName:
    timestamp = "timestamp"
    mntime = "mntime"
    errtime = "errtime"


class Columns:
    """
        数据表的列名
    """
    f_to_delete = ["changedirection"]
    p_to_delete = ["wtid"]


class PData:
    data: pd.DataFrame
    min_time: str
    max_time: str


def temporal_data_process():
    for p_name in rval_history_table_names:
        # 判断当前风机时间序列是否已经导出
        if os.path.exists(p_df_path.format(p_name)):
            continue

        logger.info("{}_df 导出中..".format(p_name))

        p_df = sql_methods \
            .select_db_to_df(p_name, sql_methods.get_all_col(p_name), order_by=ColName.mntime) \
            .rename(columns={ColName.mntime: ColName.timestamp}) \
            .drop_duplicates(subset=ColName.timestamp)

        f_df = sql_methods \
            .select_db_to_df(f_name, sql_methods.get_all_col(f_name), order_by=ColName.errtime) \
            .rename(columns={ColName.errtime: ColName.timestamp}) \
            .drop_duplicates()

        f_df = sql_methods.del_col(f_df, Columns.f_to_delete)

        f_df = f_df \
            .where(f_df["WTID"] == "I01001") \
            .dropna() \
            .reset_index()

        # 故障数据可能会出现一个时间戳有多个故障种类同时发生: 用独热编码解决

        failure_labels = sql_methods.get_col_all_value_types(f_desc_name, "ItemID")
        for label in failure_labels:
            p_df[label] = 0

        for i in tqdm(range(len(f_df))):
            p_df.loc[
                p_df[p_df[ColName.timestamp] == f_df.loc[i, ColName.timestamp]].index, f_df.loc[i, "ItemID"]] = 1

        p_df = sql_methods.del_col(p_df, Columns.p_to_delete)

        p_df.to_csv(p_df_path.format(p_name))
        logger.info("{}_df 导出成功".format(p_name))
