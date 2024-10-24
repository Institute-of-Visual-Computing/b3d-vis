using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RectPositionResetter : MonoBehaviour
{
    public RectTransform rectTransform;
    public Vector3 position;
    public bool localSpace = true;

    private void Awake()
    {
        if (!rectTransform)
        {
            rectTransform = GetComponent<RectTransform>();
        }
    }

    private void Start()
    {
        if(!rectTransform)
        {
            rectTransform = GetComponent<RectTransform>();
        }
    }

    public void ResetPosition()
    {
        if(localSpace)
        {
            rectTransform.localPosition = position;
        }
        else
        {
            rectTransform.position = position;
        }
    }
}
