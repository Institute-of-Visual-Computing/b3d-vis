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

	Task<Tuple<string, string>> fileRequestTask;

	public ServerFileCache serverFileCache;
	public UnityActionFitsNvdbRenderer nvdbRendererAction;
	public Project SelectedProject
	{
		get { return selectedProject_; }
		set
		{
			selectedProject_ = value;
			if(fileRequestTask == null || fileRequestTask.Status != TaskStatus.Running)
			{
				fileRequestTask = serverFileCache.downloadFile(selectedProject_.requests[0].result.nanoResult.resultFile);
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
	}

	private void Update()
	{
		if(fileRequestTask != null && fileRequestTask.Status == TaskStatus.RanToCompletion && fileRequestTask.IsCompletedSuccessfully)
		{
			
			var (uuid, filePath) = fileRequestTask.Result;
			if (selectedProject_.requests[0].result.nanoResult.resultFile != uuid)
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

