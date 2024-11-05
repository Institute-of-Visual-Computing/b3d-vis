using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MenuHider : MonoBehaviour
{
	public GameObject[] menus;

	public Vector3[] visiblePositions;

	public bool visiblePositionIsLocal = true;

	public void Hide(int i)
	{
		menus[i].transform.position = Camera.main.transform.position - Vector3.forward;
	}

	public void HideAll()
	{
		HideAllButShow(-1);
	}

	public void HideAllButShow(int x)
	{
		for(int i = 0; i < menus.Length; i++)
		{
			if (i != x)
			{
				menus[i].transform.position = Camera.main.transform.position - Vector3.forward;
			}
			else
			{
				if (visiblePositionIsLocal)
				{
					menus[i].transform.localPosition = visiblePositions[i % visiblePositions.Length];
				}
				else
				{
					menus[i].transform.position = visiblePositions[i % visiblePositions.Length];
				}
			}
		}
	}

	public void Show(int i)
	{
		if (visiblePositionIsLocal)
		{
			menus[i].transform.localPosition = visiblePositions[i % visiblePositions.Length];
		}
		else
		{
			menus[i].transform.position = visiblePositions[i % visiblePositions.Length];
		}
	}

	private void Start()
	{
		foreach (GameObject menu in menus)
		{
			menu.transform.position = Camera.main.transform.position - Vector3.forward;
		}
	}
}
