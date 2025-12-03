using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class RepSummaryUI : MonoBehaviour
{
    public PoseReceiver poseReceiver;

    public TextMeshProUGUI repsText;
    public TextMeshProUGUI goodText;
    public TextMeshProUGUI badText;
    public TextMeshProUGUI commonMistakeText;

    private int totalReps = 0;
    private int goodReps = 0;
    private int badReps = 0;

    private int lastRawRep = -1;   
    private bool initialized = false;

    private Dictionary<string, int> mistakeCounts = new Dictionary<string, int>();

    void Update()
    {
        PoseMessage pose = poseReceiver.GetLatestPose();
        if (pose == null)
            return;

        int rawCount = pose.rep_count;

        // First time we see a pose, just initialize baseline
        if (!initialized)
        {
            lastRawRep = rawCount;
            initialized = true;
            return;
        }

        // If Python resets rep_count (e.g. exercise change / script restart)
        if (rawCount < lastRawRep)
        {
            lastRawRep = rawCount;
            return;
        }

        int delta = rawCount - lastRawRep;
        if (delta <= 0)
            return;

        // We have new reps since last frame
        lastRawRep = rawCount;
        totalReps += delta;

        if (pose.posture_label == "good")
        {
            goodReps += delta;
        }
        else
        {
            badReps += delta;

            if (!string.IsNullOrEmpty(pose.feedback))
            {
                if (!mistakeCounts.ContainsKey(pose.feedback))
                    mistakeCounts[pose.feedback] = 0;
                mistakeCounts[pose.feedback] += delta;
            }
        }

        UpdateTexts();
    }

    void UpdateTexts()
    {
        if (repsText != null)
            repsText.text = $"Reps: {totalReps}";

        if (goodText != null)
            goodText.text = $"# of Goods: {goodReps}";

        if (badText != null)
            badText.text = $"# of Bads: {badReps}";

        if (commonMistakeText != null)
        {
            string top = GetTopMistake();
            commonMistakeText.text = string.IsNullOrEmpty(top)
                ? "Common Mistake: (none yet)"
                : "Common Mistake:\n" + top;
        }
    }

    string GetTopMistake()
    {
        int bestCount = 0;
        string bestText = null;
        foreach (var kv in mistakeCounts)
        {
            if (kv.Value > bestCount)
            {
                bestCount = kv.Value;
                bestText = kv.Key;
            }
        }
        return bestText;
    }

    public void ClearReps()
    {
       
        totalReps = 0;
        goodReps = 0;
        badReps = 0;
        mistakeCounts.Clear();

        UpdateTexts();
    }
}
