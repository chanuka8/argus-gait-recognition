from collections import Counter
from pathlib import Path
import csv

from pipeline.inference_pipeline import InferencePipeline


class FolderRecognitionPipeline:
    def __init__(self) -> None:
        self.pipeline = InferencePipeline(threshold=0.0)

    def _collect_images(
        self,
        folder_path: str,
    ) -> list[Path]:
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        if not folder.is_dir():
            raise NotADirectoryError(
                f"Not a folder: {folder_path}"
            )

        images: list[Path] = []

        for pattern in (
            "*.png",
            "*.jpg",
            "*.jpeg",
        ):
            images.extend(
                folder.glob(pattern)
            )

        return sorted(images)

    def _save_csv(
        self,
        results: list[dict],
        output_path: str,
    ) -> None:
        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image",
                    "identity",
                    "score",
                    "threshold",
                    "accepted",
                ],
            )

            writer.writeheader()

            for row in results:
                writer.writerow(row)

    def run(
        self,
        folder_path: str,
        threshold: float = 0.70,
        output_path: str | None = None,
    ) -> dict:
        self.pipeline.matcher.threshold = threshold
        images = self._collect_images(
            folder_path
        )

        if not images:
            return {
                "success": False,
                "folder": folder_path,
                "message": "No image files found",
                "threshold": threshold,
                "total": 0,
                "accepted_total": 0,
                "rejected_total": 0,
                "most_common_identity": "UNKNOWN",
                "identity_counts": {},
                "results": [],
            }

        results: list[dict] = []

        print("\n=== ARGUS FOLDER RECOGNITION ===")
        print(f"Folder: {folder_path}")
        print(f"Images found: {len(images)}")
        print(f"Threshold: {threshold:.4f}\n")

        for image_path in images:
            prediction = self.pipeline.predict(
                str(image_path)
            )

            score = round(
                float(prediction["score"]),
                4,
            )

            identity = str(
                prediction["identity"]
            )

            accepted = score >= threshold

            row = {
                "image": image_path.name,
                "identity": identity if accepted else "UNKNOWN",
                "score": score,
                "threshold": round(threshold, 4),
                "accepted": accepted,
            }

            results.append(row)

            status = "ACCEPTED" if accepted else "REJECTED"

            print(
                f"{row['image']} -> "
                f"{row['identity']} | "
                f"{row['score']:.4f} | "
                f"{status}"
            )

        accepted_results = [
            row
            for row in results
            if row["accepted"]
        ]

        identities = [
            str(row["identity"])
            for row in accepted_results
        ]

        counts = Counter(
            identities
        )

        most_common_identity = (
            counts.most_common(1)[0][0]
            if counts
            else "UNKNOWN"
        )

        summary = {
            "success": True,
            "folder": folder_path,
            "threshold": round(threshold, 4),
            "total": len(results),
            "accepted_total": len(accepted_results),
            "rejected_total": len(results) - len(accepted_results),
            "most_common_identity": most_common_identity,
            "identity_counts": dict(counts),
            "results": results,
        }

        if output_path:
            self._save_csv(
                results,
                output_path,
            )

            print(
                f"\nReport saved -> {output_path}"
            )

        print("\n=== SUMMARY ===")
        print(f"Total images: {summary['total']}")
        print(f"Accepted: {summary['accepted_total']}")
        print(f"Rejected: {summary['rejected_total']}")
        print(
            "Most common identity: "
            f"{summary['most_common_identity']}"
        )
        print(
            "Identity counts: "
            f"{summary['identity_counts']}"
        )

        return summary