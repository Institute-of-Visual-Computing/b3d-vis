using B3D;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ProjectDetails : MonoBehaviour
{
	private Project _project;

	public TextMeshProUGUI title;
	public TextMeshProUGUI fileName;
	public TextMeshProUGUI dimensions;
	public TextMeshProUGUI axisType;
	
	public Project project {
		set
		{
			this._project = value;// Set the project details
			if(_project == null)
			{
				title.text = "No Project Selected";
				fileName.text = "";
				dimensions.text = "";
				axisType.text = "";
				return;
			}

			//title.text = _project.projectName;
			fileName.text = _project.fitsOriginFileName;
			dimensions.text = string.Join(" X ", _project.fitsOriginProperties.axisDimensions);
			axisType.text = string.Join(" / ", _project.fitsOriginProperties.axisTypes);
		}
		get
		{
			return this._project;
		}
	}

	public void deselect()
	{

	}

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
