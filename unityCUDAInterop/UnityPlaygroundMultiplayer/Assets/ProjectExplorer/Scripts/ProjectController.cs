using B3D;
using MixedReality.Toolkit;
using MixedReality.Toolkit.UX;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using Unity.Netcode;
using UnityEngine.Rendering;

public class ProjectController : NetworkBehaviour
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

	public NetworkVariable<FixedString64Bytes> networkedSelectedProjectUUID = new(
		"",
		NetworkVariableReadPermission.Everyone,
		NetworkVariableWritePermission.Owner
	);

	public NetworkVariable<FixedString64Bytes> networkedRequestUUID = new(
		"",
		NetworkVariableReadPermission.Everyone,
		NetworkVariableWritePermission.Owner
	);

	public Project SelectedProject
	{
		get { return selectedProject_; }
		set
		{
			bool projectChanged = selectedProject_ != null && value != null && selectedProject_.projectUUID != value.projectUUID;
			if (IsOwner)
			{
				networkedSelectedProjectUUID.Value = new FixedString64Bytes(value == null ? "" : value.projectUUID);
			}

			if (value != null && nvdbRendererAction != null)
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
			if (IsOwner)
			{
				networkedRequestUUID.Value = new FixedString64Bytes(value == null ? "" : value.uuid);
			}

			if (selectedRequest_ != null)
			{
				if(fileRequestTask == null || fileRequestTask.Status == TaskStatus.RanToCompletion || fileRequestTask.Status == TaskStatus.Canceled)
				{
					fileRequestTask = serverFileCache.downloadFile(selectedRequest_.result.nanoResult.resultFile);
				}
			}
		}
	}

	public override void OnNetworkSpawn()
	{
		var projectUUIDString = networkedSelectedProjectUUID.Value.ToString();
		Debug.Log($"Spawned with projectUUID: {projectUUIDString}");
		if(projectUUIDString == null || projectUUIDString.Length < 1)
		{
			return;
		}
		NetworkProjectUUIDChanged(new FixedString64Bytes(""), networkedSelectedProjectUUID.Value);

		var requestUUIDString = networkedRequestUUID.Value.ToString();
		if(requestUUIDString != selectedRequest_.uuid)
		{
			Debug.Log($"Change requestUUID");
			NetworkRequestUUIDChanged(new FixedString64Bytes(""), networkedRequestUUID.Value);
		}
	}

	void Start()
	{
		serverClient.ProjectsUpdatedEvent += ProjectsUpdatedEventHandler;

		networkedSelectedProjectUUID.OnValueChanged += NetworkProjectUUIDChanged;
		networkedRequestUUID.OnValueChanged += NetworkRequestUUIDChanged;
	}

	public void NetworkRequestUUIDChanged(FixedString64Bytes previousUUID, FixedString64Bytes requestUUID)
	{
		if (IsOwner)
		{
			return;
		}
		var requestUUIDString = requestUUID.Value.ToString();
		if(requestUUIDString == selectedRequest_.uuid)
		{
			return;
		}

		if(requestUUIDString != null && requestUUIDString.Length > 0)
		{
			requestView.selectRequestExtern(requestUUIDString);
		}

	}

	public void NetworkProjectUUIDChanged(FixedString64Bytes previousUUID, FixedString64Bytes projectUUID)
	{
		if(IsOwner)
		{
			return;
		}
		
		var projectUUIDString = projectUUID.Value.ToString();
		Debug.Log($"NetworkProjectUUIDChanged: {projectUUIDString}");
		if (projectUUIDString != null && projectUUIDString.Length > 0)
		{
			var foundProjectIdx = Projects.projects.FindIndex(project => project.projectUUID == projectUUIDString);
			Debug.Log($"Found a project: {foundProjectIdx}");
			Project p = null;

			if(foundProjectIdx != -1)
			{
				p = Projects.projects[foundProjectIdx];
			}
			SelectedProject = p;
		}

		if (projectsView != null)
		{
			projectsView.selectProjectForView(projectUUIDString);
		}
	}




	override public void OnDestroy()
	{
		serverClient.ProjectsUpdatedEvent -= ProjectsUpdatedEventHandler;
		base.OnDestroy();
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

