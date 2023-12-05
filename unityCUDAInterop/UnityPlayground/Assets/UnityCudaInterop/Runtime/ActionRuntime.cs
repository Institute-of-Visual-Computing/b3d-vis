

// Assembly-CSharp, Version=0.0.0.0, Culture=neutral, PublicKeyToken=null
// ActionRuntime
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

public class ActionRuntime : MonoBehaviour
{
    public enum PluginRenderEventTypes
    {
        RTE_INITIALIZE,
        RTE_TEARDOWN,
        ACTION_REGISTER,
        ACTION_UNREGISTER,
        PluginRenderEventTypes_COUNT
    }

    private bool isInitialized = false;

    private const string dllName = "UnityCUDAInteropRuntime";

    private IntPtr _renderEventAndDataFuncPointer = IntPtr.Zero;

    private static ActionRuntime _instance;

    private CommandBuffer _immediateComandBuffer;

    private int _renderEventIDOffset = 0;

    public IntPtr RenderEventAndDataFuncPointer => _renderEventAndDataFuncPointer;

    public static ActionRuntime Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<ActionRuntime>();
                if (_instance == null)
                {
					var actionRuntimeGameObject = new GameObject("ActionRuntime");
					actionRuntimeGameObject.transform.position = Vector3.zero;
					_instance = actionRuntimeGameObject.AddComponent<ActionRuntime>();
					DontDestroyOnLoad(actionRuntimeGameObject);
                }
                
            }

            if (!_instance.isInitialized)
            {
	            _instance.InitializeRuntime();
			}
            return _instance;
        }
        private set
        {
            _instance = value;
        }
    }

    public int RenderEventIDOffset => _renderEventIDOffset;

    [DllImport(dllName)]
    private static extern IntPtr GetRenderEventAndDataFunc();

    [DllImport(dllName)]
    private static extern int InitPlugin();

    [DllImport(dllName)]
    private static extern int GetRenderEventIDOffset();

    private List<Action> registredActions_ = new List<Action>();

    private void InitializeRuntime()
    {
        //IL_0032: Unknown result type (might be due to invalid IL or missing references)
        //IL_003c: Expected O, but got Unknown
        if (!isInitialized)
        {
			Debug.Log("Init Runtime");
	        registredActions_ = new List<Action>();
			InitPlugin();
            _renderEventAndDataFuncPointer = GetRenderEventAndDataFunc();
            _renderEventIDOffset = GetRenderEventIDOffset();
            Debug.Log("renderEventIDOffset: " + _renderEventIDOffset);
			_immediateComandBuffer = new CommandBuffer();
            ExecuteAndClear(_immediateComandBuffer, getRenderEventID(PluginRenderEventTypes.RTE_INITIALIZE), IntPtr.Zero);
            isInitialized = true;
        }
    }

    public int getRenderEventID(PluginRenderEventTypes eventType)
    {
        return (int)(eventType + _renderEventIDOffset);
    }

    private void ExecuteAndClear(CommandBuffer commandBuffer, int eventId, IntPtr data)
    {
        commandBuffer.IssuePluginEventAndData(_renderEventAndDataFuncPointer, eventId, data);
        
		Graphics.ExecuteCommandBuffer(commandBuffer);
		AsyncGPUReadback.WaitAllRequests();
		commandBuffer.Clear();
    }

    public void ExecuteImmediate(int eventId, IntPtr data)
    {
        _immediateComandBuffer.IssuePluginEventAndData(_renderEventAndDataFuncPointer, eventId, data);
        
		Graphics.ExecuteCommandBuffer(_immediateComandBuffer);
		AsyncGPUReadback.WaitAllRequests();
		_immediateComandBuffer.Clear();
    }

    private void OnDestroy()
    {
	    foreach (var action in registredActions_.ToArray())
	    {
		    action.TeardownAction();
	    }
        ExecuteAndClear(_immediateComandBuffer, getRenderEventID(PluginRenderEventTypes.RTE_TEARDOWN), IntPtr.Zero);
    }

    public void registerAction(Action action)
    {
        if (action.ActionPointer == IntPtr.Zero)
        {
            Debug.LogError((object)"Can't register invalid action.");
        }
        else
        {
            ExecuteAndClear(_immediateComandBuffer, getRenderEventID(PluginRenderEventTypes.ACTION_REGISTER), action.ActionPointer);
            registredActions_.Add(action);
        }
	}

    public void unregisterAction(Action action)
    {
	    if (!registredActions_.Contains(action))
	    {
		    Debug.LogError("Action is not registered.");
		}
        if (action.ActionPointer == IntPtr.Zero)
        {
            Debug.LogError("Can't unregister invalid action.");
        }
        else
        {
		    ExecuteAndClear(_immediateComandBuffer, getRenderEventID(PluginRenderEventTypes.ACTION_UNREGISTER), action.ActionPointer);	
        }

        registredActions_.Remove(action);
    }

    private void Awake()
    {
        ActionRuntime i = Instance;
		if (i != this)
        {
            Destroy(this);
        }
    }
}
