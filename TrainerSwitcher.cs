using UnityEngine;
using UnityEngine.InputSystem;   // << important

public class TrainerSwitcher : MonoBehaviour
{
    public GameObject femaleTrainer;
    public GameObject maleTrainer;

    void Start()
    {
        ShowFemale();   // default
    }

    public void ShowFemale()
    {
        if (femaleTrainer != null) femaleTrainer.SetActive(true);
        if (maleTrainer != null) maleTrainer.SetActive(false);
    }

    public void ShowMale()
    {
        if (femaleTrainer != null) femaleTrainer.SetActive(false);
        if (maleTrainer != null) maleTrainer.SetActive(true);
    }

    void Update()
    {
        // New Input System keyboard
        var kb = Keyboard.current;
        if (kb == null) return;

        if (kb.fKey.wasPressedThisFrame)
            ShowFemale();

        if (kb.mKey.wasPressedThisFrame)
            ShowMale();
    }
}
