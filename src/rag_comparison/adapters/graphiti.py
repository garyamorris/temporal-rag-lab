import os
from datetime import datetime

# Disable upstream product analytics before importing Graphiti.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.search.search_filters import (  # noqa: E402
    ComparisonOperator as Op,
)
from graphiti_core.search.search_filters import DateFilter, SearchFilters  # noqa: E402

from ..models import Evidence, Retrieval, Update, utc_now  # noqa: E402
from .base import Adapter  # noqa: E402


def temporal_filters(as_of: datetime) -> SearchFilters:
    # Outer lists are OR, inner lists are AND. Facts are valid on [valid_at, invalid_at).
    filters = SearchFilters(
        valid_at=[
            [DateFilter(date=as_of, comparison_operator=Op.less_than_equal)],
            [DateFilter(comparison_operator=Op.is_null)],
        ],
        invalid_at=[
            [DateFilter(date=as_of, comparison_operator=Op.greater_than)],
            [DateFilter(comparison_operator=Op.is_null)],
        ],
    )
    return filters


class GraphitiAdapter(Adapter):
    async def _open(self):
        gateway = self.gateway

        class SharedEmbedder(EmbedderClient):
            def __init__(self):
                self.config = EmbedderConfig(embedding_dim=gateway.settings.embedding_dim)

            async def create(self, input_data):
                texts = [input_data] if isinstance(input_data, str) else list(input_data)
                if len(texts) != 1 or not isinstance(texts[0], str):
                    raise TypeError("Graphiti create() expects one text or a one-item text list.")
                return (await gateway.embed(texts))[0].tolist()

            async def create_batch(self, input_data_list):
                return (await gateway.embed(input_data_list)).tolist()

        config = LLMConfig(
            api_key=self.settings.openrouter_api_key.get_secret_value(),
            base_url=self.settings.openrouter_base_url,
            model=self.settings.model,
            small_model=self.settings.model,
            temperature=0,
        )
        self.graph = Graphiti(
            uri=self.settings.neo4j_uri,
            user=self.settings.neo4j_user,
            password=self.settings.neo4j_password.get_secret_value(),
            llm_client=OpenAIGenericClient(
                config=config, client=gateway.client, structured_output_mode="json_object"
            ),
            embedder=SharedEmbedder(),
            cross_encoder=OpenAIRerankerClient(config=config, client=gateway.client),
            max_coroutines=self.settings.concurrency,
        )
        await self.graph.build_indices_and_constraints()

    async def _ingest(self, update: Update) -> dict:
        previous = self.state["documents"].get(update.document_id, [])
        if update.operation == "delete":
            for episode_id in previous:
                await self.graph.remove_episode(episode_id)
            self.state["documents"][update.document_id] = []
            return {"episodes_removed": len(previous)}
        result = await self.graph.add_episode(
            name=update.id,
            episode_body=update.source_text(),
            source=EpisodeType.text,
            source_description=update.document_id,
            reference_time=update.effective_at,
            group_id=self.group_id,
        )
        episode_id = result.episode.uuid
        self.state["documents"][update.document_id] = list(dict.fromkeys([*previous, episode_id]))
        return {"episode_id": episode_id, "nodes": len(result.nodes), "edges": len(result.edges)}

    async def _retrieve(self, query: str, top_k: int, as_of: datetime | None) -> Retrieval:
        edges = await self.graph.search(
            query,
            group_ids=[self.group_id],
            num_results=top_k,
            search_filter=temporal_filters(as_of or utc_now()),
        )
        evidence = [
            Evidence(
                id=edge.uuid,
                text=edge.fact,
                kind="fact",
                source_ids=edge.episodes,
                valid_at=edge.valid_at,
                invalid_at=edge.invalid_at,
                created_at=edge.created_at,
                expired_at=edge.expired_at,
                metadata={
                    "relation": edge.name,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                },
            )
            for edge in edges
        ]
        return Retrieval(
            backend=self.name,
            query=query,
            as_of=as_of,
            native_temporal_filter=True,
            evidence=evidence,
        )

    async def close(self):
        try:
            if hasattr(self, "graph"):
                await self.graph.close()
        finally:
            await super().close()
