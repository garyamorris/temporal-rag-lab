from ..config import Settings
from ..models import BACKENDS, BackendName
from ..provider import Gateway
from .base import Adapter


def make_adapter(
    name: BackendName, namespace: str, settings: Settings, gateway: Gateway
) -> Adapter:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    if name == "graphiti":
        from .graphiti import GraphitiAdapter

        return GraphitiAdapter(name, namespace, settings, gateway)
    from .lightrag import LightRAGAdapter

    return LightRAGAdapter(name, namespace, settings, gateway)
