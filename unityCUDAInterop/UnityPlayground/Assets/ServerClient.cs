using B3D;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

[ExecuteInEditMode]
public class ServerClient : MonoBehaviour
{
	// In Editor only
	public bool getProjectsRequest = false;

	public string clientAddress = "localhost";
	public int clientPort = 5051;

	public delegate void ProjectsUpdatedEventHandler(Projects projects);
	public event ProjectsUpdatedEventHandler ProjectsUpdatedEvent;


	private bool getProjectsIsRunning = false;

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
