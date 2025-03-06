from neo4j import GraphDatabase
import pandas as pd

def import_to_neo4j(uri, username, password):
    # 连接Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # 读取CSV文件
    entities_df = pd.read_csv('entities3.csv')
    relations_df = pd.read_csv('relations3.csv')
    
    with driver.session() as session:
        # 清除现有数据
        session.run("MATCH (n) DETACH DELETE n")
        
        # 创建实体
        for _, row in entities_df.iterrows():
            cypher = """
            CREATE (n:`%s` {id: $id, name: $name})
            """ % row['实体类型']
            session.run(cypher, id=row['实体ID'], name=row['实体名称'])
        
        # 创建关系
        for _, row in relations_df.iterrows():
            cypher = """
            MATCH (a), (b)
            WHERE a.id = $head_id AND b.id = $tail_id
            CREATE (a)-[r:`%s` {category: $category}]->(b)
            """ % row['关系类型']
            session.run(cypher, 
                       head_id=row['头实体ID'],
                       tail_id=row['尾实体ID'],
                       category=row['关系类别'])
    
    driver.close()

# 使用示例
uri = "neo4j+s://9a72e3e0.databases.neo4j.io"  # Neo4j服务器地址
username = "neo4j"              # 用户名
password = "wBykwkeh9HdyspnuRAX-1DJ789CKbrDB9W4cjlRJy8U"      # 密码

import_to_neo4j(uri, username, password)