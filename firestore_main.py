from datetime import datetime, timezone
from google.cloud import firestore
from pathlib import Path
import json
import os

# Make sure env vars are set:
# export GOOGLE_APPLICATION_CREDENTIALS="/home/saimrehman/Main/headset/core-ridge-475315-q9-a6474e529dea.json"
# export GCP_PROJECT_ID="core-ridge-475315-q9"

project_id = os.environ["GCP_PROJECT_ID"]
db = firestore.Client(project=project_id)


def write_summary_to_firestore(user_id: str, summary_path: str) -> None:
    """
    Read one *_summary.json and write it as
    users/{user_id}/sessions/{video_id} in Firestore.
    """
    summary_path = Path(summary_path)

    with summary_path.open("r") as f:
        data = json.load(f)

    video_id = data["video_id"]                     # e.g. "session_20251204_003423"
    ended_at = data.get("ended_at")                # unix seconds
    if ended_at is not None:
        session_time = datetime.fromtimestamp(ended_at, tz=timezone.utc)
    else:
        session_time = datetime.now(timezone.utc)

    # Build Firestore doc from your JSON
    doc = {
        "session_time": session_time,
        "video_id": video_id,
        "exercises_seen": data.get("exercises_seen", []),
        "total_frames": data.get("total_frames"),
        "good_frames": data.get("good_frames"),
        "bad_frames": data.get("bad_frames"),
        "good_form_ratio": data.get("good_form_ratio"),
        "common_mistakes": data.get("common_mistakes", []),
        "video_uri": data.get("video_path"),   # keep name consistent if you change it
    }

    # Write: users/{user_id}/sessions/{video_id}
    (
        db.collection("users")
          .document(user_id)
          .collection("sessions")
          .document(video_id)
          .set(doc)
    )

    print(f"Wrote users/{user_id}/sessions/{video_id}")


if __name__ == "__main__":
    # Example for a single file
    user_id = "user_1"
    write_summary_to_firestore(
        user_id,
        "sessions/session_20251204_003423_summary.json",
    )
