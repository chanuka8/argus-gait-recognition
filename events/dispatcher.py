from events.event_bus import EventBus


class EventDispatcher:

    def __init__(self):

        self.bus = EventBus()

    def bus_instance(self):

        return self.bus