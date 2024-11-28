using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using B3D;
using MixedReality.Toolkit.UX;
using MixedReality.Toolkit;
using TMPro;
using Unity.XR.CoreUtils;
using System;

public class RequestsView : MonoBehaviour
{
	private Project selectedProject_;
	private Request selectedRequest_;

	public GameObject requestButtonPrefab;

	public GameObject noProjectsObject;

	public ToggleCollection toggleCollection;
	public ProjectController projectController;

	int lastSelectedButtonIndex = -1;
	int currentSelectedButtonIndex = -1;

	public Project SelectedProject
	{
		get { return selectedProject_; }
		set
		{
			selectedProject_ = value;
			if (selectedProject_ != null)
			{
				if (selectedRequest_ != null)
				{
					var currentUUID = selectedRequest_.uuid;
					selectedRequest_ = null;
					foreach (var request in selectedProject_.requests)
					{
						if (currentUUID == request.uuid)
						{
							selectedRequest_ = request;
							break;
						}
					}
				}
				if (selectedRequest_ == null)
				{
					selectedRequest_ = selectedProject_.requests[0];
					projectController.SelectedRequest = selectedRequest_;
				}
			}
			buildProjectsView();
		}
	}

	private void buildProjectsView()
	{
		List<StatefulInteractable> newToggles = new();

		foreach (Transform child in transform)
		{
			if (child.GetComponent<PressableButton>() != null)
			{
				Destroy(child.gameObject);
			}
		}

		if (selectedProject_ != null && selectedProject_.requests.Count > 0)
		{
			noProjectsObject.SetActive(false);
		}
		else
		{
			noProjectsObject.SetActive(true);
			toggleCollection.Toggles = newToggles;
			// projectDetails.project = null;
			return;
		}

		int selectedBtn = 0;
		var dummy = new GameObject("dummybtn");
		var dummyInteractable = dummy.AddComponent<StatefulInteractable>();
		newToggles.Add(dummyInteractable);
		dummy.transform.SetParent(transform);


		TextMeshProUGUI tmPro = requestButtonPrefab.GetNamedChild("Frontplate").GetNamedChild("AnimatedContent").GetNamedChild("Text").GetComponent<TextMeshProUGUI>();
		int idx = 1;
		foreach (Request request in selectedProject_.requests)
		{
			if (request.uuid == selectedRequest_.uuid)
			{
				selectedBtn = idx;
			}
			tmPro.text = request.uuid;
			GameObject projectButton = Instantiate(requestButtonPrefab, transform);
			var pbtn = projectButton.GetComponent<PressableButton>();
			newToggles.Add(pbtn);
			if (!request.result.wasSuccess() || !request.result.nanoResult.fileAvailable)
			{
				tmPro.text = "N/A " + request.uuid;
				pbtn.enabled = false;
			}
			idx++;
		}

		toggleCollection.Toggles = newToggles;
		
		toggleCollection.Toggles[selectedBtn].ForceSetToggled(true, true);
		toggleCollection.SetSelection(selectedBtn, true);
	}

	private void requestIndexSelected(int index)
	{
		if (index < 1)
		{
			return;
		}

		lastSelectedButtonIndex = currentSelectedButtonIndex;
		currentSelectedButtonIndex = index - 1;

		selectedRequest_ = selectedProject_.requests[currentSelectedButtonIndex];

		if (projectController != null)
		{
			projectController.SelectedRequest = selectedRequest_;
		}
	}

	public void selectRequestExtern(string requestUUID)
	{
		var requestIndex = selectedProject_.requests.FindIndex(request => request.uuid == requestUUID);
		toggleCollection.SetSelection(requestIndex + 1, true);
	}


	void Start()
	{
		toggleCollection.OnToggleSelected.AddListener(requestIndexSelected);
	}
}
