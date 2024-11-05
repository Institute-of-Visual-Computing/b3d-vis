using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UIElements;
using UnityEngine.UI;
using Slider = UnityEngine.UI.Slider;

public class ValueDisplay : MonoBehaviour
{
    public MixedReality.Toolkit.UX.Slider slider;
    public TextMeshProUGUI text;
    // Start is called before the first frame update
    void Start()
    {
        //slider.onValueChanged.AddListener(delegate {changed(slider.value); });
    }

    // Update is called once per frame
    void Update()
    {
        float rounded= math.round(slider.Value*100.0f)*0.01f;
        text.text = rounded.ToString();
    }

    void changed(float value)
    {
        //text.text = value.ToString();
    }
}
