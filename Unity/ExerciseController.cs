using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.Playables;

public class ExerciseController : MonoBehaviour
{
    [Header("Timeline director")]
    public PlayableDirector director;

    [Header("Exercise timelines")]
    public PlayableAsset bicepCurls;
    public PlayableAsset frontRaises;
    public PlayableAsset frontPunch;
    public PlayableAsset meleeSwing;

    [Header("Avatars")]
    public Transform woman;
    public Transform man;

    [Header("Python connection")]
    public string pythonIP = "192.168.0.10";  // <-- set this in Inspector to your LAPTOP IP
    public int pythonPort = 5006;             // must match UDP_CONTROL_PORT in Python

    private Vector3 womanStartPos;
    private Quaternion womanStartRot;
    private Vector3 manStartPos;
    private Quaternion manStartRot;

    private UdpClient udpClient;

    public string CurrentExerciseName { get; private set; }

    [Serializable]
    private class ExerciseMsg
    {
        public string exercise;
    }

    private void Awake()
    {
        if (woman != null)
        {
            womanStartPos = woman.position;
            womanStartRot = woman.rotation;
        }

        if (man != null)
        {
            manStartPos = man.position;
            manStartRot = man.rotation;
        }

        udpClient = new UdpClient();
    }

    private void OnDestroy()
    {
        try { udpClient?.Close(); }
        catch { }
    }

    void ResetAvatarsPose()
    {
        if (woman != null)
        {
            woman.position = womanStartPos;
            woman.rotation = womanStartRot;
        }

        if (man != null)
        {
            man.position = manStartPos;
            man.rotation = manStartRot;
        }
    }

    void Play(PlayableAsset asset)
    {
        if (director == null || asset == null) return;

        ResetAvatarsPose();

        director.Stop();
        director.playableAsset = asset;
        director.time = 0;
        director.RebuildGraph();
        director.Play();
    }

    // void SendExerciseToPython(string exerciseId)
    // {
    //     if (string.IsNullOrEmpty(pythonIP))
    //     {
    //         Debug.LogWarning("[ExerciseController] pythonIP not set.");
    //         return;
    //     }

    //     ExerciseMsg msg = new ExerciseMsg { exercise = exerciseId };
    //     string json = JsonUtility.ToJson(msg);
    //     byte[] data = Encoding.UTF8.GetBytes(json);

    //     try
    //     {
    //         udpClient.Send(data, data.Length, pythonIP, pythonPort);
    //         Debug.Log($"[ExerciseController] Sent {json} to {pythonIP}:{pythonPort}");
    //     }
    //     catch (Exception e)
    //     {
    //         Debug.LogError("[ExerciseController] UDP send failed: " + e.Message);
    //     }
    // }

    void SendExerciseToPython(string exerciseId)
{
    Debug.Log($"[ExerciseController] ABOUT TO SEND exercise='{exerciseId}' to {pythonIP}:{pythonPort}");

    if (string.IsNullOrEmpty(pythonIP))
    {
        Debug.LogWarning("[ExerciseController] pythonIP not set.");
        return;
    }

    ExerciseMsg msg = new ExerciseMsg { exercise = exerciseId };
    string json = JsonUtility.ToJson(msg);
    byte[] data = Encoding.UTF8.GetBytes(json);

    try
    {
        udpClient.Send(data, data.Length, pythonIP, pythonPort);
        Debug.Log($"[ExerciseController] SENT {json} to {pythonIP}:{pythonPort}");
    }
    catch (Exception e)
    {
        Debug.LogError("[ExerciseController] UDP send failed: " + e.Message);
    }
}


    // --------- Button APIs ---------

    public void PlayBicepCurls()
    {
        CurrentExerciseName = "BicepCurl";
        Play(bicepCurls);

        // Python expects "curl"
        SendExerciseToPython("curl");
    }

    public void PlayFrontRaises()
    {
        CurrentExerciseName = "FrontRaise";
        Play(frontRaises);

        // Python expects "front_raise"
        SendExerciseToPython("front_raise");
    }

    public void PlayFrontPunch()
    {
        CurrentExerciseName = "FrontPunch";
        Play(frontPunch);

        // Python expects "front_punch"
        SendExerciseToPython("front_punch");
    }

    public void PlayMeleeSwing()
    {
        CurrentExerciseName = "MeleeSwing";
        Play(meleeSwing);

        // Python expects "melee_swing"
        SendExerciseToPython("melee_swing");
    }
}
