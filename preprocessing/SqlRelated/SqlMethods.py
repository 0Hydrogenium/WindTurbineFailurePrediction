import datetime

import numpy as np
import pandas as pd

from preprocessing.SqlRelated.SqlDriver import SqlDriver


class SqlMethods:
    sql_driver = SqlDriver()

    def get_datetime_interval(self, table_name: str):
        sql = "select {} from {} order by {} ASC".format("mntime", table_name, "mntime")
        sql_results = pd.DataFrame(self.sql_driver.execute_read(sql))
        shifted_sql_results = sql_results

        if sql_results.empty:
            return 0

        shifted_sql_results = shifted_sql_results.drop(index=(len(sql_results)-1))
        shifted_sql_results = pd.concat(
            [
                pd.DataFrame([
                    sql_results.iloc[0, 0] -
                    datetime.timedelta(seconds=10)
                ]),
                shifted_sql_results
            ],
            ignore_index=True
        )

        time_delta = sql_results - shifted_sql_results - datetime.timedelta(seconds=10)

        time_delta_list = []
        for i in range(len(time_delta)):
            seconds = time_delta.iloc[i, 0].total_seconds()
            if abs(seconds) > 1:
                time_delta_list.append((i, seconds))

        return time_delta_list

    def select_db_to_df(self, table_name: str, col_list: list, order_by, num=None) -> pd.DataFrame:
        """
            将特定的database数据表中特定列的数据转换为Dataframe
        """
        complete_cols_str = ""
        for i, col in enumerate(col_list):
            complete_cols_str += str(col)
            complete_cols_str += ", " if i < len(col_list) - 1 else ""

        if num:
            sql = "select {} from {} limit {} order by {}".format(complete_cols_str, table_name, num, order_by)
        else:
            sql = "select {} from {} order by {}".format(complete_cols_str, table_name, order_by)


        results_df = pd.DataFrame(self.sql_driver.execute_read(sql), columns=col_list)

        return results_df

    def get_max_and_min(self, table_name: str, col_list: list):
        complex_col_str = ""
        for i, col in enumerate(col_list):
            complex_col_str += str(col)
            complex_col_str += ", " if i < len(col_list) - 1 else ""

        sql = "select {} from {}".format(complex_col_str, table_name)
        results_df = pd.DataFrame(self.sql_driver.execute_read(sql), columns=col_list)

        if results_df.empty:
            return 0, 0

        return max(results_df.loc[:, col_list[0]]), min(results_df.loc[:, col_list[0]])

    def get_all_col(self, table_name: str):
        """
            获取特定数据表所有的列名
        """
        sql = "select COLUMN_NAME from information_schema.COLUMNS where TABLE_NAME = '{}'".format(table_name)

        return [col[0] for col in self.sql_driver.execute_read(sql)]

    def del_col(self, df: pd.DataFrame, col_list: list):
        """
            删除某一列
        """
        return df.drop(columns=col_list)

    def get_all_table_name(self):
        sql = "select TABLE_NAME from information_schema.TABLES where TABLE_SCHEMA = 'windturbines'"

        return [col[0] for col in self.sql_driver.execute_read(sql)]

    def get_max_and_min_time(self, table_name):
        """
            获取某个风机SCADA数据的开始和结束时间
        """
        sql = "select max(mntime) as max, min(mntime) as min from {}".format(table_name)

        return [x for x in self.sql_driver.execute_read(sql)[0]]

    def get_col_all_value_types(self, table_name, col_name):
        """
            获取某个数据表的某列的所有可能出现的值
        """
        sql = "select {} from {}".format(col_name, table_name)

        return np.unique([col[0] for col in self.sql_driver.execute_read(sql)]).tolist()

