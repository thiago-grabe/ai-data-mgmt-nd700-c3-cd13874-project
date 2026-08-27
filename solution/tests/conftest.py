"""Test harness.

The hard constraint: chat.py performs Key Vault I/O at module import time, so a
bare `import chat` would trigger a device-code prompt. Every fake therefore has to
be installed before the import happens.

chat.py uses `from X import Y`, which binds the name at chat's import time. That
means patching the SOURCE module attribute (azure.keyvault.secrets.SecretClient)
works, while patching chat.SecretClient does not — the latter does not exist yet.

The whole suite runs with zero network calls.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import pytest

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))

# The exact secret names chat.py requests, in the order it requests them.
SECRET_NAMES = [
    "structuredazureendpoint",
    "structuredazureapikey",
    "structuredpostgresqlpassword",
    "structuredpostgresqluser",
    "structuredpostgresqldbname",
    "structuredpostgresqlhost",
    "unstructuredmongourl",
    "unstructureddbname",
    "unstructuredcollectionname",
    "unstructuredazureendpoint",
    "unstructuredazurekey",
    "multimodalazureconnstring",
    "multimodalazurecontentsafetyendpoint",
    "multimodalazurecontentsafetykey",
    "timeseriesazureconnstring",
    "timeseriestablename",
]

FAKE_SECRETS = {name: f"fake-{name}" for name in SECRET_NAMES}
FAKE_SECRETS["structuredpostgresqlhost"] = "fake.postgres.database.azure.com"
FAKE_SECRETS["structuredpostgresqldbname"] = "neighborhoods"
FAKE_SECRETS["structuredpostgresqluser"] = "nbhdadmin"
# Deliberately full of characters that must be percent-encoded, so a missing
# quote_plus shows up as a failing assertion rather than a production incident.
FAKE_SECRETS["structuredpostgresqlpassword"] = "p@ss/word+1"
FAKE_SECRETS["unstructuredmongourl"] = (
    "mongodb://acct:key@acct.mongo.cosmos.azure.com:10255/?ssl=true&retrywrites=false"
)
FAKE_SECRETS["multimodalazureconnstring"] = (
    "DefaultEndpointsProtocol=https;AccountName=stnbhdimg4hbr;AccountKey=aaaa==;"
    "EndpointSuffix=core.windows.net"
)
FAKE_SECRETS["timeseriesazureconnstring"] = (
    "DefaultEndpointsProtocol=https;AccountName=stnbhdsensors4hbr;AccountKey=bbbb==;"
    "EndpointSuffix=core.windows.net"
)
FAKE_SECRETS["timeseriestablename"] = "SensorReadings"


class FakeSecret:
    def __init__(self, value):
        self.value = value


class FakeSecretClient:
    """Duck-types SecretClient and records exactly what was asked for."""

    instances: ClassVar[list] = []

    def __init__(self, vault_url, credential):
        self.vault_url = vault_url
        self.credential = credential
        self.requested = []
        FakeSecretClient.instances.append(self)

    def get_secret(self, name):
        self.requested.append(name)
        # A KeyError here means chat.py asked for a name that provisioning never
        # creates — exactly the failure this fake exists to catch.
        return FakeSecret(FAKE_SECRETS[name])


@pytest.fixture(scope="session")
def chat_module():
    """Import chat.py with every Azure boundary faked out.

    The patches are active only for the duration of the import, then unwound.
    That matters: chat.py uses `from X import Y`, so the mocks stay bound inside
    chat's own namespace once imported, while the real classes are restored on
    the agent modules — letting the per-agent tests construct genuine objects.
    Holding the patches open for the whole session would hand those tests mocks.
    """
    from contextlib import ExitStack

    FakeSecretClient.instances.clear()

    with ExitStack() as stack:
        cred = stack.enter_context(
            mock.patch("azure.identity.DeviceCodeCredential", autospec=True)
        )
        stack.enter_context(
            mock.patch("azure.keyvault.secrets.SecretClient", FakeSecretClient)
        )
        agent_mocks = {
            name: stack.enter_context(mock.patch(target, autospec=True))
            for name, target in [
                ("structured", "agents.structured_data_agent.StructuredDataAgent"),
                ("unstructured", "agents.unstructured_data_agent.UnstructuredDataAgent"),
                ("multimodal", "agents.multimodal_data_agent.MultimodalDataAgent"),
                ("timeseries", "agents.timeseries_data_agent.TimeSeriesDataAgent"),
            ]
        }

        sys.modules.pop("chat", None)
        module = importlib.import_module("chat")

    module._fakes = SimpleNamespace(
        credential=cred,
        kv=FakeSecretClient.instances[-1],
        **agent_mocks,
    )
    yield module


@pytest.fixture
def manager(chat_module, tmp_path):
    """A constructed AgentChatManager with the LLM router disabled.

    Routing tests want the deterministic keyword path; the LLM router has its own
    dedicated tests.
    """
    for m in (
        chat_module._fakes.structured,
        chat_module._fakes.unstructured,
        chat_module._fakes.multimodal,
        chat_module._fakes.timeseries,
    ):
        m.reset_mock()

    with mock.patch.object(chat_module, "AuditLog") as audit_cls:
        audit_cls.return_value = mock.MagicMock()
        mgr = chat_module.AgentChatManager(use_llm_router=False)
    return mgr


@pytest.fixture
def solution_dir():
    return SOLUTION
