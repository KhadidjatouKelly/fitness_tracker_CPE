using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[System.Serializable]
public class PoseAngles
{
    public float left_elbow;
    public float right_elbow;
    public float left_shoulder;
    public float right_shoulder;
}

[System.Serializable]
public class PoseMessage
{
    public float timestamp;
    public string exercise;        // "curl", "front_raise", etc.
    public string posture_label;   // "good" or "needs_correction"
    public string feedback;        // text from Python
    public int rep_count;          // total reps for that exercise
    public PoseAngles angles;
}

public class PoseReceiver : MonoBehaviour
{
    public int port = 5005;

    private UdpClient udpClient;
    private Thread receiveThread;

    private PoseMessage latestPose;
    private object lockObj = new object();

    void Start()
    {
        udpClient = new UdpClient(port);
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveLoop()
    {
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

        while (true)
        {
            try
            {
                byte[] data = udpClient.Receive(ref anyIP);
                string json = Encoding.UTF8.GetString(data);

                PoseMessage msg = JsonUtility.FromJson<PoseMessage>(json);

                lock (lockObj)
                {
                    latestPose = msg;
                }

                // DEBUG: see what headset is actually receiving
                Debug.Log($"[PoseReceiver] got: ex={msg.exercise} label={msg.posture_label} reps={msg.rep_count}");
            }
            catch (System.Exception e)
            {
                Debug.Log("UDP Receive error: " + e.Message);
            }
        }
    }

    public PoseMessage GetLatestPose()
    {
        lock (lockObj)
        {
            return latestPose;
        }
    }

    void OnDestroy()
    {
        try
        {
            receiveThread?.Abort();
            udpClient?.Close();
        }
        catch { }
    }
}
