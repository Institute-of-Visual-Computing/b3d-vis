using B3D;
using ParrelSync;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SocialPlatforms.Impl;

public class ServerClient : MonoBehaviour
{
	// In Editor only
	public bool getProjectsRequest = false;

	public string clientAddress = "localhost";
	public int clientPort = 5051;

	public delegate void ProjectsUpdatedEventHandler(Projects projects);
	public event ProjectsUpdatedEventHandler ProjectsUpdatedEvent;

	public delegate void FileDownloadedEventHandler(string uuid, string path);
	public event FileDownloadedEventHandler FileDownloadedEvent;


	private bool getProjectsIsRunning = false;
	private bool downloadIsRunning = false;

	void setConnectionSettings(string newClientAddress, int newClientPort)
	{
		clientAddress = newClientAddress;
		clientPort = newClientPort;
	}

	IEnumerator GetProjects()
	{
		if (getProjectsIsRunning)
		{
			yield break;
		}
		getProjectsIsRunning = true;

		UnityWebRequest getProjectsRequest = UnityWebRequest.Get("http://" + clientAddress + ":" + clientPort + "/projects");
		yield return getProjectsRequest.SendWebRequest();

		if(getProjectsRequest.result != UnityWebRequest.Result.Success)
		{
			Debug.Log(getProjectsRequest.error);
		}
		else
		{
			Debug.Log(getProjectsRequest.downloadHandler.text);
			string inputJson = "{\"projects\":" + getProjectsRequest.downloadHandler.text + "}";
			Projects projs = JsonUtility.FromJson<Projects>(inputJson);
			ProjectsUpdatedEvent?.Invoke(projs);
		}
		getProjectsIsRunning = false;
	}

	IEnumerator GetFile(TaskCompletionSource<Tuple<string, string>>  promise, string destinationDirectory,  string fileUID)
	{
		if (downloadIsRunning)
		{
			yield break;
		}
		downloadIsRunning = true;
		var uwr = UnityWebRequest.Get("http://" + clientAddress + ":" + clientPort + "/file/" + fileUID);

		string path = Path.Combine(destinationDirectory, fileUID);
		#if UNITY_EDITOR
		if (ClonesManager.IsClone())
		{
			path = Path.Combine(destinationDirectory, ClonesManager.GetCurrentProject().name, fileUID);
		}
		#endif
		uwr.downloadHandler = new DownloadHandlerFile(path);
		yield return uwr.SendWebRequest();
		if (uwr.result != UnityWebRequest.Result.Success)
		{
			Debug.LogError(uwr.error);
			promise.TrySetResult(Tuple.Create(fileUID,""));
		}
		else
		{
			Debug.Log("File successfully downloaded and saved to " + path);
			FileDownloadedEvent?.Invoke(fileUID, path);
			promise.TrySetResult(Tuple.Create(fileUID, path));
		}
		downloadIsRunning = false;
	}

	public Task<Tuple<string, string>> downloadFile(string destinationDirectory, string fileUID)
	{
		var promise = new TaskCompletionSource<Tuple<string, string>>();
		if (!downloadIsRunning)
		{
			StartCoroutine(GetFile(promise, destinationDirectory, fileUID));
		} else
		{
			promise.SetCanceled();
		}
		return promise.Task;
	}

	public void getProjects()
	{
		if(!getProjectsIsRunning)
		{
			StartCoroutine(GetProjects());
		}
	}
	
	// Start is called before the first frame update
	void Start()
    {
		getProjects();
	}

	private void OnValidate()
	{
		if (getProjectsRequest)
		{
			StartCoroutine(GetProjects());
			getProjectsRequest = false;
		}
	}

	// Update is called once per frame
	void Update()
    {
        
    }
}
