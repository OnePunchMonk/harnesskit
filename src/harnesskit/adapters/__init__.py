from harnesskit.adapters.base import (
    AdapterCapabilities,
    HarnessAdapter,
    RunnableAgent,
    SupportFinding,
    SupportStatus,
    check_support,
    inspect_support,
)
from harnesskit.adapters.gateway import GatewayAdapter
from harnesskit.adapters.pydantic_ai_adapter import PydanticAIAdapter
from harnesskit.adapters.raw_api import RawAPIAdapter

ADAPTERS = {"raw_api": RawAPIAdapter, "pydantic_ai": PydanticAIAdapter, "gateway": GatewayAdapter}

__all__ = [
    "HarnessAdapter",
    "AdapterCapabilities",
    "RunnableAgent",
    "check_support",
    "inspect_support",
    "SupportFinding",
    "SupportStatus",
    "RawAPIAdapter",
    "PydanticAIAdapter",
    "GatewayAdapter",
    "ADAPTERS",
]
