using UnityEngine;
using TMPro;

public class FormEvaluator : MonoBehaviour
{
    public PoseReceiver poseReceiver;
    public TextMeshProUGUI feedbackText;

    void Update()
    {
        PoseMessage pose = poseReceiver.GetLatestPose();
        if (pose == null || feedbackText == null)
            return;

        // Just show what Python says, for ANY exercise
        if (pose.posture_label == "good")
        {
            feedbackText.text = "✅ Good form! Keep going";
        }
        else
        {
            // If Python gave specific feedback, show that
            if (!string.IsNullOrEmpty(pose.feedback))
                feedbackText.text = "❌ " + pose.feedback;
            else
                feedbackText.text = "❌ Adjust your form.";
        }
    }
}
