from collections import Counter
from collections import deque


class PredictionSmoother:

    def __init__(
        self,
        history_size=10,
        min_stable_votes=3,
    ):
        self.history_size = history_size
        self.min_stable_votes = min_stable_votes
        self.history = {}
        self.confirmed_identities = {}

    def update(
        self,
        track_id,
        prediction,
        score=1.0,
        threshold=0.85,
    ):
        if track_id not in self.history:
            self.history[track_id] = deque(
                maxlen=self.history_size
            )
            self.confirmed_identities[track_id] = "UNKNOWN"

        self.history[track_id].append(
            (prediction, score)
        )

        votes = Counter()
        for p, s in self.history[track_id]:
            if p != "UNKNOWN" and s >= threshold:
                votes[p] += 1

        if not votes:
            self.confirmed_identities[track_id] = "UNKNOWN"
            return "UNKNOWN"

        best_pred, count = votes.most_common(1)[0]
        current_confirmed = self.confirmed_identities.get(track_id, "UNKNOWN")

        if count >= self.min_stable_votes:
            self.confirmed_identities[track_id] = best_pred
            return best_pred
        elif current_confirmed != "UNKNOWN" and votes[current_confirmed] >= 1:
            return current_confirmed
        else:
            return "UNKNOWN"