import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GALLERY_SEQS = ["nm-01", "nm-02", "nm-03", "nm-04"]
DEFAULT_PROBE_NM_SEQS = ["nm-05", "nm-06"]
DEFAULT_PROBE_BG_SEQS = ["bg-01", "bg-02"]
DEFAULT_PROBE_CL_SEQS = ["cl-01", "cl-02"]


def parse_filename_meta(image_path: Path) -> dict:
    """
    Parses a CASIA-B GEI filename.
    Format example: 001_nm-01_090.png
    """
    stem = image_path.stem
    parts = stem.split("_")
    
    subject_id = parts[0] if len(parts) >= 1 else "unknown"
    seq = parts[1] if len(parts) >= 2 else "unknown"
    angle = parts[2] if len(parts) >= 3 else "unknown"

    condition = "unknown"
    if seq.startswith("nm"):
        condition = "NM"
    elif seq.startswith("bg"):
        condition = "BG"
    elif seq.startswith("cl"):
        condition = "CL"

    return {
        "subject_id": subject_id,
        "seq": seq,
        "angle": angle,
        "condition": condition,
        "path": str(image_path.resolve()),
    }


def build_gallery_and_probe_sets(
    subjects: list[str],
    gei_root: str = "data/casia_processed/gei",
    gallery_seqs: list[str] | None = None,
    probe_seqs_by_condition: dict[str, list[str]] | None = None,
    gallery_view_filter: str | None = None,  # e.g., "090" or None for all views
) -> tuple[list[dict], list[dict]]:
    if gallery_seqs is None:
        gallery_seqs = DEFAULT_GALLERY_SEQS

    if probe_seqs_by_condition is None:
        probe_seqs_by_condition = {
            "NM": DEFAULT_PROBE_NM_SEQS,
            "BG": DEFAULT_PROBE_BG_SEQS,
            "CL": DEFAULT_PROBE_CL_SEQS,
        }

    gei_path = Path(gei_root)
    subject_set = set(subjects)

    all_probe_seqs = set()
    for seq_list in probe_seqs_by_condition.values():
        all_probe_seqs.update(seq_list)

    # Verify no overlap between gallery seqs and probe seqs
    gallery_seq_set = set(gallery_seqs)
    seq_overlap = gallery_seq_set & all_probe_seqs
    if seq_overlap:
        raise ValueError(f"Sequence overlap between gallery and probe protocols: {seq_overlap}")

    gallery_items = []
    probe_items = []

    gallery_paths = set()
    probe_paths = set()

    for subject_id in sorted(subjects):
        sub_dir = gei_path / subject_id
        if not sub_dir.exists() or not sub_dir.is_dir():
            continue

        for image_path in sorted(sub_dir.glob("*.png")):
            meta = parse_filename_meta(image_path)
            meta_path = meta["path"]

            # Subject membership check
            if meta["subject_id"] not in subject_set:
                raise ValueError(f"Sample subject {meta['subject_id']} not in allowed subject set")

            # Check if gallery candidate
            if meta["seq"] in gallery_seq_set:
                if gallery_view_filter is None or meta["angle"] == gallery_view_filter:
                    if meta_path in gallery_paths:
                        raise ValueError(f"Duplicate gallery path detected: {meta_path}")
                    gallery_paths.add(meta_path)
                    gallery_items.append(meta)

            # Check if probe candidate
            elif meta["seq"] in all_probe_seqs:
                if meta_path in probe_paths:
                    raise ValueError(f"Duplicate probe path detected: {meta_path}")
                probe_paths.add(meta_path)
                probe_items.append(meta)

    # Disjointness Assertion
    path_overlap = gallery_paths & probe_paths
    if path_overlap:
        raise ValueError(f"CRITICAL: Data leakage detected! Path overlap between gallery and probe: {len(path_overlap)} files")

    return gallery_items, probe_items


def save_gallery_probe_manifest(
    gallery_items: list[dict],
    probe_items: list[dict],
    output_path: str = "configs/gallery_probe_manifest.json",
    protocol_name: str = "Standard CASIA-B (Gallery: NM 1-4, Probe: NM 5-6, BG 1-2, CL 1-2)",
) -> None:
    manifest = {
        "protocol": protocol_name,
        "gallery_sample_count": len(gallery_items),
        "probe_sample_count": len(probe_items),
        "gallery_subjects": sorted(list(set(item["subject_id"] for item in gallery_items))),
        "probe_subjects": sorted(list(set(item["subject_id"] for item in probe_items))),
        "probe_conditions": sorted(list(set(item["condition"] for item in probe_items))),
        "gallery_sample_paths": [item["path"] for item in gallery_items],
        "probe_sample_paths": [item["path"] for item in probe_items],
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


if __name__ == "__main__":
    from evaluation.dataset_split import load_or_create_subject_split
    split = load_or_create_subject_split()
    test_subs = split["test_subjects"]
    gal, prb = build_gallery_and_probe_sets(test_subs)
    print("Test subjects gallery/probe sets built successfully:")
    print(f"Gallery samples: {len(gal)} | Probe samples: {len(prb)}")
    save_gallery_probe_manifest(gal, prb)
