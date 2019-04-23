import re
from typing import List

from elasticsearch import Elasticsearch


class EsHit:
    def __init__(self, score: float, position: int, text: str, type: str):
        """
        Basic information about an ElasticSearch Hit
        :param score: score returned by the query
        :param position: position in the retrieved results (before any filters are applied)
        :param text: retrieved sentence
        :param type: type of the hit in the index (by default, only documents of type "sentence"
        will be retrieved from the index)
        """
        self.score = score
        self.position = position
        self.text = text
        self.type = type


class EsSearch:
    @staticmethod
    def construct_qa_query(max_hits_retrieved, random_seed=1234):
        return {"from": 0, "size": max_hits_retrieved,
                "query": {
                    "function_score": {
                        "query": {
                            "match_all": {}
                        },
                        "random_score": {
                            # set a seed to get same random text
                            "seed": random_seed
                        }
                    }
                }}

    # Constructs an ElasticSearch query from the input question and choice
    # Uses the last self._max_question_length characters from the question and
    # requires that the
    def __init__(self,
                 es_client: str = "localhost",
                 indices: str = "arc_corpus",
                 random_seed: int = 1234):
        """
        Class to search over the text corpus using ElasticSearch
        :param es_client: Location of the ElasticSearch service
        :param indices: Comma-separated list of indices to search over
        """
        self._es = Elasticsearch([es_client], retries=3)
        self._indices = indices
        self.random_seed = random_seed
        # Regex for negation words used to ignore Lucene results with negation
        self._negation_regexes = [re.compile(r) for r in
                                  ["not\\s", "n't\\s", "except\\s"]]

    # text matches the answer choice and the hit type is a "sentence"

    # This limites are on words, not characters
    def get_hits(self,
                 max_hits_retrieved: int = 100,
                 max_filtered_hits: int = 30,
                 max_hit_length: int = 100,
                 min_hit_length: int = 10):
        res = self._es.search(
            index=self._indices,
            body=self.construct_qa_query(max_hits_retrieved, self.random_seed)
        )
        hits = []
        for idx, es_hit in enumerate(res['hits']['hits']):
            es_hit = EsHit(score=es_hit['_score'],
                           position=idx,
                           text=es_hit['_source']['text'],
                           type=es_hit['_type'])
            hits.append(es_hit)
        return self.filter_hits(hits, max_filtered_hits, max_hit_length,
                                min_hit_length)

    # Remove hits that contain negation, are too long, are duplicates,
    # are noisy.
    def filter_hits(self, hits: List[EsHit], max_filtered_hits: int,
                    max_hit_length: int, min_hit_length: int) -> List[
        EsHit]:
        filtered_hits = []
        selected_hit_keys = set()
        for hit in hits:
            hit_sentence = hit.text
            hit_sentence = hit_sentence.strip().replace("\n", " ")
            words = hit_sentence.split()
            if len(words) > max_hit_length or len(words) < min_hit_length:
                continue
            for negation_regex in self._negation_regexes:
                if negation_regex.search(hit_sentence):
                    # ignore hit
                    continue
            if self.get_key(hit_sentence) in selected_hit_keys:
                continue
            if not self.is_clean_sentence(hit_sentence):
                continue
            filtered_hits.append(hit)
            selected_hit_keys.add(self.get_key(hit_sentence))
        return filtered_hits[:max_filtered_hits]

    # Check if the sentence is not noisy
    @staticmethod
    def is_clean_sentence(s):
        # must only contain expected characters, should be single-sentence and
        # only uses hyphens
        # for hyphenated words
        return (re.match("^[a-zA-Z0-9][a-zA-Z0-9;:,\(\)%\-\&\.'\"\s]+\.?$",
                         s) and
                not re.match(".*\D\. \D.*", s) and
                not re.match(".*\s\-\s.*", s))

    # Create a de-duplication key for a HIT
    @staticmethod
    def get_key(hit):
        # Ignore characters that do not effect semantics of a sentence and URLs
        return re.sub('[^0-9a-zA-Z\.\-^;&%]+', '',
                      re.sub('http[^ ]+', '', hit)).strip().rstrip(".")


if __name__ == '__main__':
    es_search = EsSearch()
    res = es_search.get_hits(max_hits_retrieved=100, max_filtered_hits=10,
                             max_hit_length=50, min_hit_length=5)
    for hit in res:
        print(hit.text)
