using B3D;
using MixedReality.Toolkit;
using MixedReality.Toolkit.UX;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

public class ProjectController : MonoBehaviour
{
	public B3D.Projects Projects { get; set; }

	public PressableButton projectDetailSelectButton;

	public ProjectsView projectsView;

	public ServerClient serverClient;

	private Project selectedProject_;
	private Request selectedRequest_;

	public RequestsView requestView;

	Task<Tuple<string, string>> fileRequestTask;

	public ServerFileCache serverFileCache;
	public UnityActionFitsNvdbRenderer nvdbRendererAction;
	public Project SelectedProject
	{
		get { return selectedProject_; }
		set
		{
			bool projectChanged = selectedProject_ != null && value != null && selectedProject_.projectUUID != value.projectUUID;
			if(value != null)
			{
				nvdbRendererAction.objectRenderer.enabled = true;
				nvdbRendererAction.objectRenderer.transform.Find("Coordinates").gameObject.SetActive(true);
			}
			selectedProject_ = value;
			if (requestView)
			{
				requestView.SelectedProject = value;
				
			}
			
		}
	}

	public Request SelectedRequest
	{
		get => selectedRequest_;
		set
		{
			selectedRequest_ = value;
			if(selectedRequest_ != null)
			{
				if(fileRequestTask == null || fileRequestTask.Status == TaskStatus.RanToCompletion || fileRequestTask.Status == TaskStatus.Canceled)
				{
					fileRequestTask = serverFileCache.downloadFile(selectedRequest_.result.nanoResult.resultFile);
				}
			}
		}
	}


	void Start()
	{
		serverClient.ProjectsUpdatedEvent += ProjectsUpdatedEventHandler;
	}

	private void OnDestroy()
	{
		serverClient.ProjectsUpdatedEvent -= ProjectsUpdatedEventHandler;
	}

	protected void ProjectsUpdatedEventHandler(Projects newProjects)
	{
		projectsView.Projects = newProjects;
		Projects = newProjects;
		if(selectedProject_ != null)
		{
			foreach (var project in newProjects.projects)
			{
				if(selectedProject_.projectUUID == project.projectUUID)
				{
					SelectedProject = project;
					break;
				}
			}
		}
	}

	private void Update()
	{
		if(fileRequestTask != null && fileRequestTask.Status == TaskStatus.RanToCompletion && fileRequestTask.IsCompletedSuccessfully)
		{
			
			var (uuid, filePath) = fileRequestTask.Result;
			if (selectedRequest_.result.nanoResult.resultFile != uuid)
			{
				return;
			}
			fileRequestTask = null;
			Debug.Log($"File with uuid: {uuid} downloaded to {filePath}");
			if(nvdbRendererAction != null)
			{
				nvdbRendererAction.showVolume(selectedProject_, uuid);
			}
		}
	}
}

