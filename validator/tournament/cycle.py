import asyncio

from validator.core.config import load_config
from validator.tournament.tournament_manager import process_active_tournaments
from validator.tournament.tournament_manager import process_pending_rounds
from validator.tournament.tournament_manager import process_pending_tournaments
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections


logger = get_logger(__name__)


async def cycle():
    config = load_config()

    await try_db_connections(config)

    await asyncio.gather(
        # this gets the submissions and populates the tournament participants
        process_pending_tournaments(config),
        # this processes pending rounds by creating tasks and assigning nodes
        process_pending_rounds(config),
        # this advances the tournament till completion
        process_active_tournaments(config),
    )


if __name__ == "__main__":
    asyncio.run(cycle())
