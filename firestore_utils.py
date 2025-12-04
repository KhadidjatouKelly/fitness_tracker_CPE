# # firestore_utils.py

from datetime import datetime, timezone
from google.cloud import firestore
from google.api_core.exceptions import DeadlineExceeded
import time
import os

project_id = os.environ["GCP_PROJECT_ID"]
db = firestore.Client(project=project_id)


def push_session_to_firestore(user_id: str, summary: dict) -> None:
    video_id = summary["video_id"]
    ended_at = summary.get("ended_at")

    if ended_at is not None:
        session_time = datetime.fromtimestamp(ended_at, tz=timezone.utc)
    else:
        session_time = datetime.now(timezone.utc)

    doc = {
        "session_time": session_time,
        "video_id": video_id,
        "exercises_seen": summary.get("exercises_seen", []),
        "total_frames": summary.get("total_frames"),
        "good_frames": summary.get("good_frames"),
        "bad_frames": summary.get("bad_frames"),
        "good_form_ratio": summary.get("good_form_ratio"),
        "common_mistakes": summary.get("common_mistakes", []),
        "video_uri": summary.get("video_path"),
    }

    doc_ref = (
        db.collection("users")
          .document(user_id)
          .collection("sessions")
          .document(video_id)
    )

    # Simple manual retry: 3 attempts with a longer timeout
    for attempt in range(3):
        try:
            doc_ref.set(doc, timeout=30)  # ← increased timeout
            print(f"[Firestore] Wrote users/{user_id}/sessions/{video_id}")
            return
        except DeadlineExceeded as e:
            print(f"[Firestore] DeadlineExceeded on attempt {attempt+1}: {e}")
            if attempt == 2:
                print("[Firestore] Giving up after 3 attempts.")
                raise
            time.sleep(2)  # small backoff before retry

# from datetime import datetime, timezone
# from google.cloud import firestore
# import os

# project_id = os.environ["GCP_PROJECT_ID"]
# db = firestore.Client(project=project_id)


# def push_session_to_firestore(user_id: str, summary: dict) -> None:
#     """
#     summary is exactly the dict you showed:
#     {
#       "video_id": "...",
#       "exercises_seen": [...],
#       "total_frames": ...,
#       "good_frames": ...,
#       "bad_frames": ...,
#       "good_form_ratio": ...,
#       "common_mistakes": [...],
#       "video_path": "...",
#       "ended_at": 1764794426.811969,
#       ...
#     }
#     """
#     video_id = summary["video_id"]
#     ended_at = summary.get("ended_at")

#     # use ended_at as the session timestamp if present
#     if ended_at is not None:
#         session_time = datetime.fromtimestamp(ended_at, tz=timezone.utc)
#     else:
#         session_time = datetime.now(timezone.utc)

#     doc = {
#         "session_time": session_time,
#         "video_id": video_id,
#         "exercises_seen": summary.get("exercises_seen", []),
#         "total_frames": summary.get("total_frames"),
#         "good_frames": summary.get("good_frames"),
#         "bad_frames": summary.get("bad_frames"),
#         "good_form_ratio": summary.get("good_form_ratio"),
#         "common_mistakes": summary.get("common_mistakes", []),
#         "video_uri": summary.get("video_path"),
#     }

#     (
#         db.collection("users")
#           .document(user_id)
#           .collection("sessions")
#           .document(video_id)
#           .set(doc)
#     )

#     print(f"[Firestore] Wrote users/{user_id}/sessions/{video_id}")
