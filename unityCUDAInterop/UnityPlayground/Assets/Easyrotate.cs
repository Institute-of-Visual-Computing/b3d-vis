using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Easyrotate : MonoBehaviour
{
    public float rotationSpeed;
    public Transform targetTransform;
    public OptixInterop optixInterop;


    public float transferSpeed;

    public float startVal = 0.431f;
    public float endVal = 0.4836f;

    [Range(0f, 1f)]
    public float animationState = 0.0f;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        targetTransform.RotateAround(Vector3.up, Time.deltaTime * rotationSpeed);
        float diff = endVal - startVal;
        optixInterop.transferOffset = (Mathf.Sin(Time.time * transferSpeed) *0.5f + 0.5f) * diff + startVal;
        // 
    }
}
