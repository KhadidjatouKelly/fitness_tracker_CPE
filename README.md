# Upper-Body Fitness Tracker
**Unity VR Client + Pose Analysis Service Backend**

A VR fitness training prototype that combines:
- **VR interaction + avatar exercise demo** (Unity/OpenXR on Meta Quest 3)
- **Real-time pose estimation + rep counting + form feedback** (Python + MediaPipe)
- **UDP + JSON messaging** between the VR client and the analysis service
- **session summary upload to Firestore**

## System Overview

The project runs as two cooperating components:

1) **VR Client (Unity/OpenXR)**
- Handles UI, avatar selection (male/female), exercise selection, and in-VR feedback display.
- Sends exercise selection commands to the backend via UDP.

2) **Pose Analysis Service (Python/MediaPipe)**
- Captures webcam frames, runs MediaPipe Pose, computes joint angles, counts reps, and generates feedback.
- Sends live metrics (angles/labels/reps) back to Unity via UDP.
- Saves an annotated video + JSON summary at the end of a session and can upload the summary to Firestore.

## Repository Structure

- `Unity/` — Unity project (VR client)  
- `live_pose_to_unity.py` — main real-time pose → Unity pipeline (4 exercises)  
- `firestore_utils.py` — Firestore helper functions  
- `firestore_main.py` — Firestore-related entry/testing script (if used)  
- `session_*_summary.json` — example session summary output  

## Supported Exercises

The backend supports four exercises (selected by Unity):
- `curl`
- `front_raise`
- `front_punch`
- `melee_swing`

Unity sends:
```json
{"exercise": "curl"}
