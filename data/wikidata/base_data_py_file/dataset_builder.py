import logging
from pathlib import Path

import ir_datasets
import pandas as pd
from typing import NamedTuple, Dict, List, Optional
import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class DatasetBuilder:
    def __init__(
            self,
            docstore: NamedTuple,
            query_qid_df: pd.DataFrame,
            item_info_df: pd.DataFrame,
            adj_item_info_df: pd.DataFrame,
            triple_id_df: pd.DataFrame,
            qrels_df: Optional[pd.DataFrame] = None
    ):
        self.docstore = docstore
        self.query_qid_df = query_qid_df
        self.item_info_df = item_info_df
        self.adj_item_info_df = adj_item_info_df
        self.triple_id_df = triple_id_df
        self.qrels_df = qrels_df

    @staticmethod
    def _get_item_info(single_item_info_df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """
        Extract entity information from DataFrame.
        Get the entity information corresponding to the query
        """

        if single_item_info_df.empty:
            # logging.warning(f"No item info found for item_qid: {item_qid}")
            return None

        required_fields = ['label_zh', 'label_kk', 'description_zh', 'description_kk']
        if single_item_info_df[required_fields].isnull().any(axis=1).any():
            # logging.warning(f"Missing required fields for item_qid: {item_qid}")
            return None

        single_item_info = {
            "label_zh": single_item_info_df['label_zh'].values[0],
            "label_kk": single_item_info_df['label_kk'].values[0],
            "description_zh": single_item_info_df['description_zh'].values[0],
            "description_kk": single_item_info_df['description_kk'].values[0]
        }

        return single_item_info

    def _get_adjacent_items_info(self, q_item_qid: str, adj_item_num: int) -> Optional[Dict[str, List[str]]]:
        """Get information about adjacent items."""
        adj_item_qids = self.triple_id_df[self.triple_id_df['item_qid'] == q_item_qid]['adj_item_qid']

        if adj_item_qids.empty:
            logging.warning(f"No adjacent items found for q_item_qid: {q_item_qid}")
            return None

        # Ensure we have the correct number of adjacent items
        # 相邻实体数量在做实验的时候 要确保 少的 相邻实体 是多的相邻实体的子集
        adj_item_qids = (adj_item_qids.sample(n=adj_item_num, replace=True)
                         if len(adj_item_qids) < adj_item_num
                         else adj_item_qids.sample(n=adj_item_num))

        adj_item_info = {
            "label_zh": [], "label_kk": [],
            "description_zh": [], "description_kk": []
        }

        for adj_item_qid in adj_item_qids:
            single_adj_item_info_df = self.adj_item_info_df[self.adj_item_info_df['item_qid'] == adj_item_qid]
            single_adj_item_info = self._get_item_info(single_adj_item_info_df)

            # 最好通过 self.adj_item_info_df 过滤一下 self.triple_id_df
            if not single_adj_item_info:
                return None

            adj_item_info["label_zh"].append(single_adj_item_info.get('label_zh'))
            adj_item_info["label_kk"].append(single_adj_item_info.get('label_kk'))
            adj_item_info["description_zh"].append(single_adj_item_info.get('description_zh'))
            adj_item_info["description_kk"].append(single_adj_item_info.get('description_kk'))

        return adj_item_info

    def _get_documents(self, query_id: str, pos_doc_num: int, neg_doc_num: int) -> Optional[Dict[str, List[str]]]:
        """Get positive and negative documents for training data."""
        if self.qrels_df is None:
            return None

        query_docs = self.qrels_df[self.qrels_df['query_id'] == query_id]
        if query_docs.empty:
            logging.warning(f"No relevant documents found for query_id: {query_id}")
            return None

        try:
            pos_doc_ids = query_docs[query_docs['relevance'] != 0]['doc_id'].head(pos_doc_num)
            neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'].sample(n=neg_doc_num)

            return {
                "pos_doc": [self.docstore.get(doc_id).text for doc_id in pos_doc_ids],
                "neg_doc": [self.docstore.get(doc_id).text for doc_id in neg_doc_ids]
            }
        except Exception as e:
            logging.error(f"Error getting documents for query_id {query_id}: {e}")
            return None

    def build(self, output_file: str, adj_item_num: int = 3,
              dataset_type: str = 'train', pos_doc_num: int = 1,
              neg_doc_num: int = 1) -> None:
        """Build and save the dataset."""
        if dataset_type not in ['train', 'test']:
            raise ValueError("dataset_type must be either 'train' or 'test'")

        jsonl_data = []

        for _, row in tqdm(self.query_qid_df.iterrows(),
                           total=self.query_qid_df.shape[0],
                           desc=f"Building {dataset_type} dataset"):
            try:
                query_id = row['query_id']
                query_text = row['query_text']
                q_item_qid = row['q_item_qid']

                # Get query entity information
                q_item_info_df = self.item_info_df[self.item_info_df['item_qid'] == q_item_qid]
                q_item_info = self._get_item_info(q_item_info_df)
                if not q_item_info:
                    continue

                # Get adjacent items information
                adj_item_info = self._get_adjacent_items_info(q_item_qid, adj_item_num)
                if not adj_item_info:
                    continue

                # Prepare base data
                dataset_item = {
                    "query_id": query_id,
                    "q_item_qid": q_item_qid,
                    "query": query_text,
                    "q_item_info": q_item_info,
                    "adj_item_info": adj_item_info,
                }

                # Add documents for training data
                if dataset_type == 'train':
                    doc_data = self._get_documents(query_id, pos_doc_num, neg_doc_num)
                    if not doc_data:
                        continue
                    dataset_item.update(doc_data)

                jsonl_data.append(dataset_item)

            except Exception as e:
                logging.error(f"Error processing row {row}: {e}")
                continue

        # Write data to JSONL file
        with jsonlines.open(output_file, mode='w') as writer:
            writer.write_all(jsonl_data)

        logging.info(f"Dataset built and saved to {output_file}. Data size: {len(jsonl_data)}")


def main(
        docstore: NamedTuple,
        query_qid_file: str,
        qrels_file: str,
        item_info_file: str,
        adj_item_info_file: str,
        triple_id_file: str,
        output_file: str,
        adj_item_num: int = 3,
        dataset_type: str = 'train',
        pos_doc_num: int = 1,
        neg_doc_num: int = 1
) -> None:
    """Main function to build dataset."""
    try:
        # Load all required dataframes
        query_qid_df = pd.read_csv(query_qid_file, encoding='utf-8').astype(str)
        item_info_df = pd.read_csv(item_info_file, encoding='utf-8').astype(str)
        adj_item_info_df = pd.read_csv(adj_item_info_file, encoding='utf-8').astype(str)
        triple_id_df = pd.read_csv(triple_id_file, encoding='utf-8').astype(str)

        qrels_df = None
        if dataset_type == 'train':
            qrels_df = pd.read_csv(qrels_file, encoding='utf-8')
            qrels_df['query_id'] = qrels_df['query_id'].astype(str)
            qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
            qrels_df['relevance'] = qrels_df['relevance'].astype(int)

        # Create dataset builder instance
        builder = DatasetBuilder(
            docstore=docstore,
            query_qid_df=query_qid_df,
            item_info_df=item_info_df,
            adj_item_info_df=adj_item_info_df,
            triple_id_df=triple_id_df,
            qrels_df=qrels_df
        )

        # Build dataset
        builder.build(
            output_file=output_file,
            adj_item_num=adj_item_num,
            dataset_type=dataset_type,
            pos_doc_num=pos_doc_num,
            neg_doc_num=neg_doc_num
        )

    except Exception as e:
        logging.error(f"Error building dataset: {e}")
        raise


if __name__ == "__main__":
    HOME_DIR = Path(__file__).parent.parent / 'base_data_file'
    ADJ_ITEM_NUM = 3
    POS_DOC_NUM = 1
    NEG_DOC_NUM = 1

    # 加载 zh-kk clir 数据集
    CLIRMatrix_dataset_train = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
    # 加载原始查询数据
    # queries_df = pd.DataFrame(CLIRMatrix_dataset.queries_iter())
    # 加载 doc 数据
    docs_docstore = CLIRMatrix_dataset_train.docs_store()
    # 加载原始的 qrels 数据
    # train_qrels_df = pd.DataFrame(CLIRMatrix_dataset_train.qrels_iter())

    # -------------------- 构建 train dataset 文件 --------------------
    # main(
    #     docstore=docs_docstore,
    #     query_qid_file=str(HOME_DIR / 'base_train_query_entity_qid_final.csv'),
    #     qrels_file=str(HOME_DIR / 'base_train_qrels.csv'),
    #     item_info_file=str(HOME_DIR / 'base_train_query_entity_info_filled.csv'),
    #     adj_item_info_file=str(HOME_DIR / 'base_train_adj_item_info_filled.csv'),
    #     triple_id_file=str(HOME_DIR / 'base_train_triplet_id_fragment_3_final.csv'),
    #     output_file=str(HOME_DIR / 'train_dataset.jsonl'),
    #     adj_item_num=ADJ_ITEM_NUM,
    #     dataset_type='train',
    #     pos_doc_num=POS_DOC_NUM,
    #     neg_doc_num=NEG_DOC_NUM
    # )

    # -------------------- 构建 test dataset 文件 --------------------
    main(
        docstore=docs_docstore,
        query_qid_file=str(HOME_DIR / 'base_test_query_entity_qid_final.csv'),
        qrels_file=str(HOME_DIR / 'base_test_qrels.csv'),
        item_info_file=str(HOME_DIR / 'base_test_query_entity_info_filled.csv'),
        adj_item_info_file=str(HOME_DIR / 'base_test_adj_item_info_filled.csv'),
        triple_id_file=str(HOME_DIR / 'base_test_triplet_id_final.csv'),
        output_file=str(HOME_DIR / 'test_dataset.jsonl'),
        adj_item_num=ADJ_ITEM_NUM,
        dataset_type='test',
        pos_doc_num=POS_DOC_NUM,
        neg_doc_num=NEG_DOC_NUM
    )
    pass
