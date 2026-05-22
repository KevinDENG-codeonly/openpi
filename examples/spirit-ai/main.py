import dataclasses
import importlib
import logging

from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import action_chunk_resample_broker
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

_env = importlib.import_module("examples.spirit-ai.env")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    realsense_serials: str = "230322270398,313522302626,230422271253"
    camera_resolutions: str = "320*240,320*240,320*240"
    structure: str = "wholebody"
    prompt: str = "fold the paper box"

    action_horizon: int = 40
    resample_ratio: float = 4.0

    num_episodes: int = 1
    max_episode_steps: int = 2000


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    runtime = _runtime.Runtime(
        environment=_env.SpiritaiMoz1Environment(
            realsense_serials=args.realsense_serials,
            camera_resolutions=args.camera_resolutions,
            structure=args.structure,
            prompt=args.prompt,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_resample_broker.ActionChunkResampleBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                resample_ratio=args.resample_ratio,
            )
        ),
        subscribers=[],
        max_hz=120,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
