"""Harbor Modal environment with safe dynamic-network bootstrapping."""

from __future__ import annotations

from typing import Any

from harbor.environments.modal import ModalEnvironment
from harbor.models.task.config import NetworkPolicy


class DynamicModalEnvironment(ModalEnvironment):
    """Work around Modal requiring an allowlist at sandbox creation time.

    Harbor 0.21 correctly plans a no-network baseline and an endpoint-only
    agent phase. Modal 1.5 cannot transition a sandbox created with no domain
    allowlist into allowlist mode. Create dynamic sandboxes with wildcard
    capability, then immediately restore the frozen baseline before setup.
    """

    _bootstrapping_dynamic_network = False

    def _dynamic_network_kwargs(self, network_policy: NetworkPolicy) -> dict[str, Any]:
        if self._bootstrapping_dynamic_network:
            return {
                "outbound_domain_allowlist": ["*"],
                "outbound_cidr_allowlist": ["0.0.0.0/0"],
            }
        return ModalEnvironment._dynamic_network_kwargs(network_policy)

    async def _create_sandbox(
        self,
        *,
        entrypoint: list[str] | None = None,
        block_network: bool | None = None,
        experimental_options: dict[str, Any] | None = None,
    ):
        if not self._dynamic_network:
            return await super()._create_sandbox(
                entrypoint=entrypoint,
                block_network=block_network,
                experimental_options=experimental_options,
            )
        self._bootstrapping_dynamic_network = True
        try:
            sandbox = await super()._create_sandbox(
                entrypoint=entrypoint,
                block_network=block_network,
                experimental_options=experimental_options,
            )
        finally:
            self._bootstrapping_dynamic_network = False
        await sandbox._experimental_set_outbound_network_policy.aio(
            **ModalEnvironment._dynamic_network_kwargs(self.network_policy)
        )
        return sandbox
