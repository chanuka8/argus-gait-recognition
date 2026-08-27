from pipeline.steps.feature_extraction import (
    FeatureExtractionStep,
)
from pipeline.steps.matching_step import (
    MatchingStep,
)
from storage.vector_store import (
    VectorStore,
)


class InferencePipeline:
    def __init__(
        self,
        threshold: float = 0.85,
        gallery_dir: str = "models/live_gallery",
    ):

        self.extractor = FeatureExtractionStep()

        self.matcher = MatchingStep(threshold=threshold)

        self.store = VectorStore(gallery_dir=gallery_dir)

        gallery = self.store.load()

        if gallery is None:
            raise RuntimeError("Gallery not built.")

        (
            self.gallery_features,
            self.gallery_labels,
            self.metadata,
        ) = gallery

    def predict(
        self,
        image_path,
    ):

        embedding = self.extractor.extract(image_path)

        open_set_res = self.matcher.match_open_set(
            embedding,
            self.gallery_features,
            self.gallery_labels,
            self.metadata,
        )

        identity, score = self.matcher.match(
            embedding,
            self.gallery_features,
            self.gallery_labels,
            self.metadata,
        )

        return {
            "identity": identity,
            "score": score,
            "open_set_state": open_set_res.state.value,
        }
