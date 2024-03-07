using UnityEngine;
using UnityEditor;
using UnityEngine.UIElements;
using System.Text.RegularExpressions;

public class EditorCreateAction : EditorWindow
{
	#region internal structs

	struct AssetTextReplacementInfos
	{
		public string regex;
		public string replacement;
		public RegexOptions options;
	}

	struct AssetDuplicationInfos
	{
		public string templateName;
		public string libraryName;
		public string className;
		public string actionFolderPath;
		public AssetTextReplacementInfos[] replacementInfos;
	}

	#endregion

	#region menu entries

	[MenuItem("Assets/Create/UnityCudaInterop/RenderAction", priority = 0)]
	private static void CreateRenderAction()
	{
		EditorCreateAction window = GetWindow<EditorCreateAction>(true, "Create RenderAction");

		window.position = new Rect(Screen.width / 2, Screen.height / 2, 450, 150);
		window.Show();
	}

	#endregion menu entries

	#region class properties

	const string defaultActionParentAssetFolderPath = @"UnityCudaInterop/Actions";

	const string renderActionTemplateName = "RenderActionTemplate";
	const string unityRenderActionTemplateName = "UnityRenderActionTemplate";

	const string lineToRemoveRegex = @"^.*\<LINETOREMOVE\>.*$";

	private TextField actionNameTextField;
	private Button createBtn;

	#endregion class properties

	private void CreateGUI()
	{
		var label = new Label("Create a new RenderAction for UnityCudaInterop.");

		actionNameTextField = new TextField
		{
			label = "Action Name:",
			multiline = false
		};

		createBtn = new Button
		{
			text = "Create!",
		};
		createBtn.clicked += () => tryAddAction();

		rootVisualElement.Add(label);
		rootVisualElement.Add(actionNameTextField);
		rootVisualElement.Add(createBtn);
		
	}

	private void tryAddAction()
	{
		if(actionNameTextField.text.Length > 0)
		{
			if(tryAddAction(actionNameTextField.text))
			{
				this.Close();
			}
		}
	}

	/// <summary>
	/// Add action 
	/// </summary>
	/// <param name="actionLibraryName"></param>
	/// <returns>false if folder creation fails. true otherwise.</returns>
	private bool tryAddAction(string actionLibraryName)
	{
		var actionParentFolderGUID = createFolderInAssets($@"{defaultActionParentAssetFolderPath}/{actionLibraryName}");
		if (actionParentFolderGUID.Length == 0)
		{
			Debug.LogError("Could not create required folder in assets.");
			return false;
		}

		var actionParentFolderPath = AssetDatabase.GUIDToAssetPath(actionParentFolderGUID);

		var actionClassName = actionLibraryName;
		var unityActionClassName = $"Unity{char.ToUpper(actionLibraryName[0])}{actionLibraryName[1..]}";

		AssetDuplicationInfos actionClassDuplicateInfo = new()
		{
			templateName = renderActionTemplateName,
			libraryName = actionLibraryName,
			className = actionClassName,
			actionFolderPath = actionParentFolderPath,
			replacementInfos = new AssetTextReplacementInfos[]
			{
				new()
				{
					regex = lineToRemoveRegex,
					replacement = "",
					options = RegexOptions.Multiline
				},
				new()
				{
					regex = renderActionTemplateName,
					replacement = actionLibraryName,
					options = RegexOptions.Multiline
				}
			}
		};

		AssetDuplicationInfos unityActionClassDuplicateInfo = new()
		{
			templateName = unityRenderActionTemplateName,
			libraryName = actionLibraryName,
			className = unityActionClassName,
			actionFolderPath = actionParentFolderPath,
			replacementInfos = new AssetTextReplacementInfos[]
			{
				new()
				{
					regex = lineToRemoveRegex,
					replacement = "",
					options = RegexOptions.Multiline
				},
				new()
				{
					regex = unityRenderActionTemplateName,
					replacement = unityActionClassName,
					options = RegexOptions.Multiline
				},
				new()
				{
					regex = "TemplateRenderEventTypes",
					replacement = $"{actionLibraryName}RenderEventTypes",
					options = RegexOptions.Multiline
				},
				new()
				{
					regex = "TemplateNativeRenderingData",
					replacement = $"{actionLibraryName}NativeRenderingData",
					options = RegexOptions.Multiline
				},
				new()
				{
					regex = "templateNativeRenderingData_",
					replacement =  $"{char.ToLower(actionLibraryName[0])}{actionLibraryName[1..]}NativeRenderingData_",
					options = RegexOptions.Multiline
				},
			}
		};


		string duplicatedAssetGUId = duplicateTemplateActionAsset(actionClassDuplicateInfo);
		if (duplicatedAssetGUId.Length == 0)
		{
			Debug.LogError($"Could not duplicate {actionClassDuplicateInfo.templateName} for action {actionClassDuplicateInfo.libraryName}");
		}

		duplicatedAssetGUId = duplicateTemplateActionAsset(unityActionClassDuplicateInfo);
		if (duplicatedAssetGUId.Length == 0)
		{
			Debug.LogError($"Could not duplicate {actionClassDuplicateInfo.templateName} for action {actionClassDuplicateInfo.libraryName}");
		}

		return true;
	}

	/// <summary>
	/// Duplicate an asset with given name in a folder with a new name and the new file extension.
	/// </summary>
	/// <param name="assetName">Name of the assets without file extension.</param>
	/// <param name="destinationFolderPath">folder for the newly created duplicate.</param>
	/// <param name="destinationName">nane of the duplicate</param>
	/// <param name="fileExtension">file extension of the duplicate</param>
	/// <returns>GUID of the new asset if operation succeeds. Empty string otherwise.</returns>
	private string duplicateTemplate(string assetName, string destinationFolderPath, string destinationName, string fileExtension)
	{
		// Find Template
		string[] guids = AssetDatabase.FindAssets(assetName);
		if (guids.Length == 0)
		{
			Debug.LogError($"Could not find template file {assetName}.");
			return "";
		}
		string renderActionTemplateGUID = guids[0];

		// Copy Template to new destination
		string renderActionScriptPath = $@"{destinationFolderPath}/{destinationName}.{fileExtension}";

		if (!AssetDatabase.CopyAsset(AssetDatabase.GUIDToAssetPath(renderActionTemplateGUID), renderActionScriptPath))
		{
			Debug.LogError($"Could not duplicate {assetName} to {renderActionScriptPath}");
			return "";
		}
		return AssetDatabase.AssetPathToGUID(renderActionScriptPath);
	}

	/// <summary>
	/// Replace content in a asset based in replacementinfos.
	/// </summary>
	/// <param name="assetFilePath">AssetPath to the File. e.g. Assets/path/to/file.extension</param>
	/// <param name="replacementInfos"></param>
	private void replaceContentInTextAsset(string assetFilePath, ref AssetTextReplacementInfos[] replacementInfos)
	{
		var str = System.IO.File.ReadAllText(assetFilePath);
		foreach(var replacementInfo in replacementInfos)
		{
			str = Regex.Replace(str, replacementInfo.regex, replacementInfo.replacement, replacementInfo.options);
		}
		System.IO.File.WriteAllText(assetFilePath, str);
		AssetDatabase.ImportAsset(assetFilePath);
	}

	/// <summary>
	/// duplicate template with given AssetDuplicationInfos
	/// </summary>
	/// <param name="duplicationInfo">input parameters for duplication.</param>
	/// <returns>GUID of new assets if operation succeeds. Empty string otherwise.</returns>
	private string duplicateTemplateActionAsset(AssetDuplicationInfos duplicationInfo)
	{
		var classGUIDs = AssetDatabase.FindAssets(duplicationInfo.className);
		foreach (var guid in classGUIDs)
		{
			var guidAssetPath = AssetDatabase.GUIDToAssetPath(guid);
			if (guidAssetPath.EndsWith($"{duplicationInfo.className}.cs"))
			{
				Debug.LogError($"Class for action with this name already exists at {guidAssetPath}.");
				return "";
			}
		}

		string renderActionDuplicateAssetGUID = duplicateTemplate(duplicationInfo.templateName, duplicationInfo.actionFolderPath, duplicationInfo.className, "cs");
		string renderActionScriptPath = AssetDatabase.GUIDToAssetPath(renderActionDuplicateAssetGUID);

		replaceContentInTextAsset(renderActionScriptPath, ref duplicationInfo.replacementInfos);

		return renderActionDuplicateAssetGUID;
	}
	

	/// <summary>
	/// Creates all folders in the chain under the Assets folder.
	/// </summary>
	/// <param name="pathInAssetsFolder">folder chain. Must NOT start with Assets!</param>
	/// <returns>GUID of new created folder if opertation succeeds. Empty string otherwise.</returns>
	private string createFolderInAssets(string pathInAssetsFolder)
	{
		var assetsFolderPath = @$"Assets/{pathInAssetsFolder}";
		
		if (AssetDatabase.IsValidFolder(assetsFolderPath))
		{
			return AssetDatabase.AssetPathToGUID(assetsFolderPath);
		}

		var assetPathComponentArray = assetsFolderPath.Split("/");
		var lastFolderGUID = "";
		for (int i = 1; i < assetPathComponentArray.Length; i++)
		{
			var currentFolderPath = string.Join("/", assetPathComponentArray, 0, i + 1);
			if (!AssetDatabase.IsValidFolder(currentFolderPath))
			{
				var parentFolderPath = string.Join("/", assetPathComponentArray, 0, i);
				lastFolderGUID = AssetDatabase.CreateFolder(parentFolderPath, assetPathComponentArray[i]);
				if (lastFolderGUID.Length == 0)
				{
					Debug.LogError($"Could not create folder {assetPathComponentArray[i]} in folder {parentFolderPath}");
					return lastFolderGUID;
				}
			}
		}
		return lastFolderGUID;
	}
}
