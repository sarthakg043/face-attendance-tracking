import numpy as np
from deepface import DeepFace
from config import MODEL_NAME, DETECTOR, EUCLIDEAN_THR, COSINE_THR
from database import db_get_all_persons


def extract_embedding(image_path: str) -> list[float]:
    """Extract ArcFace embedding from an image. Raises ValueError if no face found."""
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True,
        align=True,
    )
    return result[0]["embedding"]


# ── Distance helpers ──────────────────────────────────────────────────────────

def euclidean_distance(a: list, b: list) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def cosine_distance(a: list, b: list) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def _best_distance(probe_emb: list, enrolled_embeddings: list) -> tuple:
    best_euc, best_cos = float("inf"), float("inf")
    for emb in enrolled_embeddings:
        euc = euclidean_distance(probe_emb, emb)
        cos = cosine_distance(probe_emb, emb)
        if cos < best_cos:
            best_euc, best_cos = euc, cos
    return best_euc, best_cos


# ── Core match ────────────────────────────────────────────────────────────────

def match_face(probe_image_path: str, top_k: int = 3) -> dict:
    """
    1-to-N identification. Returns best_match, is_match, top_k candidates,
    and a strategy_comparison dict.
    Only considers active persons.
    """
    probe_emb = extract_embedding(probe_image_path)
    enrolled = db_get_all_persons(active_only=True)
    if not enrolled:
        return {"best_match": None, "is_match": False, "top_k": [], "strategy_comparison": {}}

    candidates = []
    for person in enrolled:
        euc, cos = _best_distance(probe_emb, person["embedding"])
        candidates.append({
            "person_id": person["id"],
            "roll_no": person["roll_no"],
            "name": person["name"],
            "department": person["department"],
            "num_gallery": len(person["embedding"]),
            "euclidean": round(euc, 4),
            "cosine": round(cos, 4),
            "euc_match": euc < EUCLIDEAN_THR,
            "cos_match": cos < COSINE_THR,
        })

    candidates.sort(key=lambda x: (x["cosine"], x["euclidean"]))
    best = candidates[0]
    is_match = best["euc_match"] and best["cos_match"]

    strategy_comparison = {
        "euclidean": {"matched": best["euc_match"], "distance": best["euclidean"], "threshold": EUCLIDEAN_THR},
        "cosine": {"matched": best["cos_match"], "distance": best["cosine"], "threshold": COSINE_THR},
        "agreement": best["euc_match"] == best["cos_match"],
        "final": "MATCH" if is_match else "NO MATCH",
    }

    return {
        "best_match": best if is_match else None,
        "closest_candidate": best,
        "is_match": is_match,
        "top_k": candidates[:top_k],
        "strategy_comparison": strategy_comparison,
    }


def verify_two_faces(path_a: str, path_b: str) -> dict:
    emb_a = extract_embedding(path_a)
    emb_b = extract_embedding(path_b)
    euc = euclidean_distance(emb_a, emb_b)
    cos = cosine_distance(emb_a, emb_b)
    return {
        "euclidean_distance": round(euc, 4),
        "cosine_distance": round(cos, 4),
        "euclidean_match": euc < EUCLIDEAN_THR,
        "cosine_match": cos < COSINE_THR,
        "is_same_person": (euc < EUCLIDEAN_THR) and (cos < COSINE_THR),
    }
