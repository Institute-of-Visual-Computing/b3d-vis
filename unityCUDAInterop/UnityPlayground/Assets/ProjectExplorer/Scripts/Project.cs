using B3D;
using System;
using System.Collections.Generic;

namespace B3D
{
	/// \brief Copied from cfitsio.h. Not to mistaken with data length in bytes.
	[Serializable]
	public enum FitsImageType : int
	{
		UNKNOWN = 0,
		BYTE = 8,
		SHORT = 16,
		INT = 32,
		LONG = 64,
		FLOAT = -32,
		DOUBLE = -64
	};

	/// \brief Copied from cfitsio.h. Not to mistaken with data length in bytes. <summary>
	/// </summary>
	[Serializable]
	public enum FitsDataTypes : int
	{
		UNKNOWN = 0,
		SBYTE = 11,
		USHORT = 20,
		SHORT = 21,
		UINT = 30,
		INT = 31,
		LONG = 41,
		ULONG = 40,
		FLOAT = 42,
		LONGLONG = 81,
		DOUBLE = 82
	};

	//// \brief Common Properties of a FITS file used in the library.
	[Serializable]
	public class FitsProperties
	{
		public int axisCount;
		public FitsImageType imgType;
		public List<long> axisDimensions;
		public List<string> axisTypes;
	};

	[Serializable]
	public class Box3I
	{
		public UnityEngine.Vector3Int lower;
		public UnityEngine.Vector3Int upper;
	};

	[Serializable]
	public class SofiaParams
	{
		public Dictionary<string, string> params_;
	};

	[Serializable]
	public class BaseResult
	{
		public int returnCode;
		public string message;
		public bool finished;

		// Timestamp of finish. Since Epoch in seconds.
		public long finishedAt;

		public bool wasSuccess() {
			return finished && returnCode == 0;
		}
	};

	[Serializable]
	public class BaseFileResult : BaseResult
	{
		// either a UUID or a path
		public string resultFile;
		public bool fileAvailable;
	};

	[Serializable]
	public class SofiaResult : BaseFileResult
	{

	};

	[Serializable]
	public class NanoResult : BaseFileResult
	{
		// Size and position of the vdb
		// Due to not existing values for empty space the vdb can be cropped to the bounding box of the data.
		// The bounding box of the original data and the vdb can be different.
		public UnityEngine.Vector3Int voxelSize;

		// Offset of the vdb in the world space with respect to the original data.
		public UnityEngine.Vector3Int voxelOffset;
	};

	[Serializable]
	public class PipelineResult : BaseResult
	{
		public SofiaResult sofiaResult;
		public NanoResult nanoResult;
	};

	[Serializable]
	public class Request
	{
		public string uuid;

		public Box3I subRegion;

		// SofiaParams
		public SofiaParams sofiaParameters;
		public PipelineResult result;

		// Timestamp of creation. Since Epoch in seconds. 
		public long createdAt;
	}

	[Serializable]
	public class Project
	{
		public string b3dViewerProjectVersion = "1.0";
		public string projectName;
		public string projectUUID;
		public string fitsOriginUUID;
		public string fitsOriginFileName;
		// TODO: Creator Name, Creation Date
		public FitsProperties fitsOriginProperties;
		public List<Request> requests;
	}


	[Serializable]
	public class Projects
	{
		public List<Project> projects;
	}

}
