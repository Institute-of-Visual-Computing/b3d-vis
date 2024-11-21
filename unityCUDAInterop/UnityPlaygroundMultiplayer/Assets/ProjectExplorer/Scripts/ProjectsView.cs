using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using B3D;
using MixedReality.Toolkit.UX;
using UnityEngine.Events;
using MixedReality.Toolkit;
using Unity.XR.CoreUtils;
using TMPro;

public class ProjectsView : MonoBehaviour
{
	public ServerClient serverClient;

	public GameObject projectButtonPrefab;

	public ToggleCollection toggleCollection;

	public GameObject noProjectsText;

	private Projects projects_;

	public ProjectDetails projectDetails;
	public ProjectController projectController;

	public Projects Projects
	{
		get { return projects_; }
		set
		{
			projects_ = value;
			buildProjectsView();
		}
	}

	private void buildProjectsView()
	{
		List<StatefulInteractable> newToggles = new();

		if (projects_.projects.Count > 0)
		{
			noProjectsText.SetActive(false);
		}
		else
		{
			noProjectsText.SetActive(true);
			toggleCollection.Toggles = newToggles;
			projectDetails.project = null;
			return;
		}

		foreach (Transform child in transform)
		{
			if (child.GetComponent<PressableButton>() != null)
			{
				Destroy(child.gameObject);
			}
		}
		int selectedBtn = 0;
		var dummy = new GameObject("dummybtn");
		var dummyInteractable = dummy.AddComponent<StatefulInteractable>();
		newToggles.Add(dummyInteractable);
		dummy.transform.SetParent(transform);

		TextMeshProUGUI tmPro = projectButtonPrefab.GetNamedChild("Frontplate").GetNamedChild("AnimatedContent").GetNamedChild("Text").GetComponent<TextMeshProUGUI>();
		int idx = 1;
		foreach (Project project in projects_.projects)
		{
			if (projectDetails.project != null && projectDetails.project.projectUUID == project.projectUUID)
			{
				selectedBtn = idx;
			}
			idx++;
			tmPro.text = project.projectName;
			GameObject projectButton = Instantiate(projectButtonPrefab, transform);
			newToggles.Add(projectButton.GetComponent<PressableButton>());
		}

		toggleCollection.Toggles = newToggles;
		toggleCollection.Toggles[selectedBtn].ForceSetToggled(true);
		toggleCollection.SetSelection(selectedBtn, true);

	}

	private void projectIndexSelected(int index)
	{
		if(index < 1)
		{
			return;
		}
		projectDetails.project = projects_.projects[index-1];
		projectController.SelectedProject = projects_.projects[index-1];
	}

	// Start is called before the first frame update
	void Start()
    {
		toggleCollection.OnToggleSelected.AddListener(projectIndexSelected);
	}
}
