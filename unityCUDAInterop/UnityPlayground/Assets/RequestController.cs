using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using B3D;

public class RequestController : MonoBehaviour
{
	private Project selectedProject_;
	private Request selectedRequest_;

	public ServerClient serverClient;

	public Project SelectedProject
	{
		get { return selectedProject_; }
		set
		{
			selectedProject_ = value;
			if(selectedRequest_ != null && selectedProject_ != null)
			{
				foreach (var request in selectedProject_.requests)
				{
					if(selectedRequest_.uuid == request.uuid)
					{
						selectedRequest_ = request;
						break;
					}
				}
			}
		}
	}
}
