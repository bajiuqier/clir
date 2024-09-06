import csv
from pathlib import Path
from py2neo import Graph, Node, Relationship

# 连接到Neo4j数据库
graph = Graph("bolt://62.234.35.209:7687", auth=("neo4j", "yh980727@"))

# 清空数据库（谨慎使用！）
graph.delete_all()
# 读取并导入实体
def import_entities(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            node = Node("Entity", id=row['item'])
            for key, value in row.items():
                if key != 'item':
                    if value is not None and value != "":
                        # 只有当值不为None且不为空字符串时才添加属性
                        node[key] = value
            graph.create(node)

# 读取并导入关系
def import_relationships(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            node = Node("Relationship", id=row['item'])
            for key, value in row.items():
                if key != 'item':
                    if value is not None and value != "":
                        # 只有当值不为None且不为空字符串时才添加属性
                        node[key] = value
            graph.create(node)

# 创建实体间的关系
def create_entity_relationships(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            entity1_id, relationship_id, entity2_id = row
            entity1 = graph.nodes.match("Entity", id=entity1_id).first()
            entity2 = graph.nodes.match("Entity", id=entity2_id).first()
            relationship = graph.nodes.match("Relationship", id=relationship_id).first()
            
            if entity1 and entity2 and relationship:
                rel = Relationship(entity1, "RELATED", entity2)
                for key, value in relationship.items():
                    rel[key] = value
                graph.create(rel)

# 主函数
def main():
    HOME_DIR = Path(__file__).parent / 'neo4j'
    import_entities(str(HOME_DIR / 'item_info.csv'))
    import_entities(str(HOME_DIR / 'adjitem_info.csv'))

    import_relationships(str(HOME_DIR / 'property_info.csv'))
    create_entity_relationships(str(HOME_DIR / 'filtered_triplet_id.csv'))

    print("Data import completed.")

if __name__ == "__main__":
    main()