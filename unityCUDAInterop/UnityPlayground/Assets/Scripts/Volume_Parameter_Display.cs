using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class Volume_Parameter_Display : MonoBehaviour
{
    public TextMeshProUGUI text1, text2, text3;

    public BoxCollider volumeCollider, cubeCollider;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        text1.text = cubeCollider.bounds.ToString();
    }
}
