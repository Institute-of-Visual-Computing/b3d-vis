using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class HidableObject
{
	public GameObject objToHide;
	public Vector3 visiblePosition;
	public bool visiblePositionIsLocal;

	public bool isVisible;
}

public class MenuHider : MonoBehaviour
{
	public HidableObject[] hidableObjects;


	public void Hide(int i)
	{
		hidableObjects[i].isVisible = false;
        hidableObjects[i].objToHide.transform.position = Camera.main.transform.position - Camera.main.transform.forward;
    }

	public void HideAll()
	{
        HideAllButShow(-1);
	}

	public void HideAllButShow(int x)
	{
		for(int i = 0; i < hidableObjects.Length; i++)
		{
			if (i != x)
			{
				Hide(i);

            }
			else
			{
				Show(i);
			}
		}
	}

	public void Show(int i)
	{
		hidableObjects[i].isVisible = true;

        if (hidableObjects[i].visiblePositionIsLocal)
		{
			hidableObjects[i].objToHide.transform.localPosition = hidableObjects[i].visiblePosition;
		}
		else
		{
            hidableObjects[i].objToHide.transform.position = hidableObjects[i].visiblePosition;
        }
	}


    private void Update()
    {
		foreach (var item in hidableObjects)
		{
            if(!item.isVisible)
			{
                item.objToHide.transform.position = Camera.main.transform.position - Camera.main.transform.forward;
            }
        }
    }
}
