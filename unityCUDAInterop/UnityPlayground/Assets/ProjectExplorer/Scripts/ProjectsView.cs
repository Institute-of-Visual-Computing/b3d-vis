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

		TextMeshProUGUI tmPro = projectButtonPrefab.GetNamedChild("Frontplate").GetNamedChild("AnimatedContent").GetNamedChild("Text").GetComponent<TextMeshProUGUI>();
		foreach (Project project in projects_.projects)
		{
			tmPro.text = project.projectName;
			GameObject projectButton = Instantiate(projectButtonPrefab, transform);
			newToggles.Add(projectButton.GetComponent<PressableButton>());
		}
		toggleCollection.Toggles = newToggles;
		toggleCollection.Toggles[0].ForceSetToggled(true);
		toggleCollection.SetSelection(0, true);
	}

	private void projectIndexSelected(int index)
	{
		projectDetails.project = projects_.projects[index];
		projectController.SelectedProject = projects_.projects[index];
	}

	// Start is called before the first frame update
	void Start()
    {
		toggleCollection.OnToggleSelected.AddListener(projectIndexSelected);
	}
}
