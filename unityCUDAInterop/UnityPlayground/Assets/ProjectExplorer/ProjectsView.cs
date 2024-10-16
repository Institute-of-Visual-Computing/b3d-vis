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

	private Projects projects;

	public ProjectDetails projectDetails;

	private void projectIndexSelected(int index)
	{
		Debug.Log("Selected");
		projectDetails.project = projects.projects[index];
	}

	// Start is called before the first frame update
	void Start()
    {
		serverClient.ProjectsUpdatedEvent += ProjectsUpdatedEventHandler;
		toggleCollection.OnToggleSelected.AddListener(projectIndexSelected);
	}


	void OnDestroy()
	{
		serverClient.ProjectsUpdatedEvent -= ProjectsUpdatedEventHandler;
	}

	protected void ProjectsUpdatedEventHandler(Projects newProjects)
	{
		projects = newProjects;
		// toggleCollection.Toggles.Clear();

		if (projects.projects.Count > 0)
		{
			noProjectsText.SetActive(false);
		}
		else
		{
			noProjectsText.SetActive(true);
		}

		List<StatefulInteractable> newToggles = new();
		TextMeshProUGUI tmPro = projectButtonPrefab.GetNamedChild("Frontplate").GetNamedChild("AnimatedContent").GetNamedChild("Text").GetComponent<TextMeshProUGUI>();
		foreach (Project project in projects.projects)
		{
			tmPro.text = project.projectName;
			GameObject projectButton = Instantiate(projectButtonPrefab, transform);
			newToggles.Add(projectButton.GetComponent<PressableButton>());
		}
		toggleCollection.Toggles = newToggles;
	}

	// Update is called once per frame
	void Update()
    {
    }
}
