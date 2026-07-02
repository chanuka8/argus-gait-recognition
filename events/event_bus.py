from collections import defaultdict
from typing import Callable

from core.logger import setup_logger


class EventBus:

    def __init__(self):

        self.logger = setup_logger("ARGUS.EventBus")

        self._subscribers = defaultdict(list)

    def subscribe(
        self,
        event_name: str,
        callback: Callable
    ) -> None:

        self._subscribers[event_name].append(callback)

        self.logger.info(
            f"Subscriber registered -> {event_name}"
        )

    def publish(
        self,
        event_name: str,
        data=None
    ) -> None:

        self.logger.info(
            f"Event published -> {event_name}"
        )

        for callback in self._subscribers[event_name]:

            try:
                callback(data)

            except Exception as ex:

                self.logger.error(
                    f"Event callback failed: {ex}"
                )