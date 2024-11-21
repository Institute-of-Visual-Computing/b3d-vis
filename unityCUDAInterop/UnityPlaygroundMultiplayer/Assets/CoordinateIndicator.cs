using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CoordinateIndicator : MonoBehaviour
{
	public GameObject xAxisObject;
	public GameObject yAxisObject;
	public GameObject zAxisObject;
	public GameObject originObject;

	[Range(0.003f, 1f)]
	[SerializeField]
	public float localLineLength = 1.0f;

	[Range(0.001f, 0.5f)]
	[SerializeField]
	private float localLineWidth = 0.02f;

	public float LocalLineLength
	{
		get
		{
			return localLineLength;
		}
		set
		{
			localLineLength = value;
			setLineLengths();
		}
	}

	public float LocalLineWidth
	{
		get
		{
			return localLineWidth;
		}
		set
		{
			localLineWidth = value;
			setScales();
			scaleAxisObject(xAxisObject);
			scaleAxisObject(yAxisObject);
			scaleAxisObject(zAxisObject);
			scaleOriginObject();
		}
	}

	private void OnValidate()
	{
		setScales();
		setLineLengths();
	}

	void scaleOriginObject()
	{
		if (originObject != null)
		{
			originObject.transform.localScale = Vector3.one * 2 * localLineWidth;
		}
	}

	void setScales()
	{
		if (xAxisObject != null)
		{
			scaleAxisObject(xAxisObject);
		}
		if (yAxisObject != null)
		{
			scaleAxisObject(yAxisObject);
		}
		if (zAxisObject != null)
		{
			scaleAxisObject(zAxisObject);
		}
		scaleOriginObject();
	}

	void scaleAxisObject(GameObject axisObject)
	{
		LineRenderer axisLineRenderer = axisObject.GetComponent<LineRenderer>();
		axisLineRenderer.startWidth = localLineWidth;
		axisLineRenderer.endWidth = localLineWidth;
	}

	void setLineLengths()
	{
		if(xAxisObject != null)
		{
			setLineLength(xAxisObject, new Vector3(1, 0, 0) * localLineLength);
		}
		if (yAxisObject != null)
		{
			setLineLength(yAxisObject, new Vector3(0, 1, 0) * localLineLength);
		}
		if (zAxisObject != null)
		{
			setLineLength(zAxisObject, new Vector3(0, 0, 1) * localLineLength);
		}
	}
	void setLineLength(GameObject axisObject, Vector3 lineRendererPoint)
	{
		LineRenderer xAxisLineRenderer = axisObject.GetComponent<LineRenderer>();
		xAxisLineRenderer.SetPosition(1, lineRendererPoint);
	}
}
