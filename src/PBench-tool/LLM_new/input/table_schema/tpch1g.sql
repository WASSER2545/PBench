CREATE TABLE region (
    r_regionkey      INTEGER NOT NULL,
    r_name           VARCHAR(25) NOT NULL,
    r_comment        VARCHAR(152),
    PRIMARY KEY (r_regionkey)
);

CREATE TABLE nation (
    n_nationkey      INTEGER NOT NULL,
    n_name           VARCHAR(25) NOT NULL,
    n_regionkey      INTEGER NOT NULL,
    n_comment        VARCHAR(152),
    PRIMARY KEY (n_nationkey),
    FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)
);

CREATE TABLE supplier (
    s_suppkey        INTEGER NOT NULL,
    s_name           VARCHAR(25) NOT NULL,
    s_address        VARCHAR(40) NOT NULL,
    s_nationkey      INTEGER NOT NULL,
    s_phone          VARCHAR(15) NOT NULL,
    s_acctbal         DECIMAL(15,2),
    s_comment        VARCHAR(101),
    PRIMARY KEY (s_suppkey),
    FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE customer (
    c_custkey        INTEGER NOT NULL,
    c_name           VARCHAR(25) NOT NULL,
    c_address        VARCHAR(40) NOT NULL,
    c_nationkey      INTEGER NOT NULL,
    c_phone          VARCHAR(15) NOT NULL,
    c_acctbal        DECIMAL(15,2),
    c_mktsegment     VARCHAR(10) NOT NULL,
    c_comment        VARCHAR(117),
    PRIMARY KEY (c_custkey),
    FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE partsupp (
    ps_partkey       INTEGER NOT NULL,
    ps_suppkey       INTEGER NOT NULL,
    ps_availqty       INTEGER,
    ps_supplycost    DECIMAL(15,2) NOT NULL,
    ps_comment       VARCHAR(199),
    PRIMARY KEY (ps_partkey, ps_suppkey),
    FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey),
    FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey)
);

CREATE TABLE part (
    p_partkey        INTEGER NOT NULL,
    p_name           VARCHAR(55) NOT NULL,
    p_mfgr           VARCHAR(25) NOT NULL,
    p_brand          VARCHAR(10) NOT NULL,
    p_type           VARCHAR(25) NOT NULL,
    p_size           INTEGER,
    p_container      VARCHAR(10) NOT NULL,
    p_retailprice    DECIMAL(15,2) NOT NULL,
    p_comment        VARCHAR(23),
    PRIMARY KEY (p_partkey)
);

CREATE TABLE orders (
    o_orderkey       INTEGER NOT NULL,
    o_custkey        INTEGER,
    o_orderstatus    VARCHAR(1) NOT NULL,
    o_totalprice     DECIMAL(15,2),
    o_orderdate      DATE,
    o_orderpriority  VARCHAR(15) NOT NULL,
    o_clerk          VARCHAR(15) NOT NULL,
    o_shippriority   INTEGER,
    o_comment        VARCHAR(79),
    PRIMARY KEY (o_orderkey),
    FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
);

CREATE TABLE lineitem (
    l_orderkey       INTEGER NOT NULL,
    l_partkey        INTEGER NOT NULL,
    l_suppkey        INTEGER NOT NULL,
    l_linenumber     INTEGER NOT NULL,
    l_quantity       DECIMAL(15,2) NOT NULL,
    l_extendedprice  DECIMAL(15,2) NOT NULL,
    l_discount       DECIMAL(15,2) NOT NULL,
    l_tax            DECIMAL(15,2) NOT NULL,
    l_returnflag     VARCHAR(1) NOT NULL,
    l_linestatus     VARCHAR(1) NOT NULL,
    l_shipdate       DATE,
    l_commitdate     DATE,
    l_receiptdate    DATE,
    l_shipinstruct   VARCHAR(25) NOT NULL,
    l_shipmode       VARCHAR(10) NOT NULL,
    l_comment        VARCHAR(44),
    PRIMARY KEY (l_orderkey, l_linenumber),
    FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
    FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)
);