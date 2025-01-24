import numpy as np


def generate_zipfian_keys(n, num_keys, max_key):
    """ Generate keys following a Zipfian distribution. """
    keys = np.random.zipf(num_keys, n)
    keys = np.mod(keys, max_key) + 1  # Ensure keys do not exceed max_key
    return keys


def create_sql_queries(range_keys, specific_keys):
    """ Create SQL queries based on the provided keys. """
    sql_queries = []

    # Generate specific field queries
    for key in specific_keys:
        sql = f"SELECT * FROM ?.usertable WHERE ycsb_key = {key};"
        sql_queries.append(sql)

    # Generate range queries
    for i in range(0, len(range_keys), 2):
        start_key, end_key = range_keys[i], range_keys[i + 1]
        if end_key < start_key:
            start_key, end_key = end_key, start_key
        sql = f"SELECT * FROM ?.usertable WHERE ycsb_key BETWEEN {start_key} AND {end_key};"
        sql_queries.append(sql)

    return sql_queries


def write_queries_to_file(sql_queries, file_path):
    """ Write the SQL queries to a file. """
    with open(file_path, 'w') as file:
        for i, sql in enumerate(sql_queries, 1):
            file.write(f" '.SQL/{i}.0'\n{sql}\n\n")


if __name__ == "__main__":
    num_queries = 10
    zipf_param = 2
    max_key_value = 1000000

    # Generate keys for queries
    range_query_keys = generate_zipfian_keys(num_queries * 2, zipf_param, max_key_value)
    specific_field_query_keys = generate_zipfian_keys(num_queries, zipf_param, max_key_value)

    # Create SQL queries
    sql_queries = create_sql_queries(range_query_keys, specific_field_query_keys)

    # Write queries to file
    file_path = 'ycsb.sql'
    write_queries_to_file(sql_queries, file_path)

    print(f"SQL queries have been written to {file_path}")

