from typing import List


class QueryEnhancementService:
    async def enhance_query(self, query: str) -> str:
        return query

    async def expand_with_synonyms(self, query: str) -> List[str]:
        return [query]

    async def generate_sub_queries(self, query: str) -> List[str]:
        return [query]
