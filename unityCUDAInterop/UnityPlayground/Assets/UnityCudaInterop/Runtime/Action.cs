
using System;
using UnityEngine;

using System.Collections;
using System.Runtime.CompilerServices;

public abstract class Action : MonoBehaviour

{
    public class ActionRenderEventIDS
    {
        public const int ACTION_INITIALIZE = 0;

        public const int ACTION_TEARDOWN = 1;

        public const int NATIVE_ACTION_RENDER_EVENT_ID_COUNT = 2;
    }

    protected ActionRuntime actionRuntime;

    private IntPtr actionPointer = IntPtr.Zero;

    private int _renderEventIdOffset = 0;

	private bool _isInitialized = false;

	public bool Isinitialized
	{
		get => _isInitialized;
		private set => _isInitialized = value;
	}

    public IntPtr ActionPointer => actionPointer;

    public int RenderEventIdOffset => _renderEventIdOffset;

    
    private IEnumerator WaitUntilActionIsRegisteredEnumerator()
    {
	    yield return new WaitUntil(() => getRenderEventIdOffset() > 0);
	    _renderEventIdOffset = getRenderEventIdOffset();
		ActionRegisterDone();
    }

    private IEnumerator InitializeAfterRegistered()
    {
	    yield return new WaitUntil(isRegistered);
		initialize();

	}

	public int getRenderEventID(int baseRenderEventId)
    {
        return _renderEventIdOffset + baseRenderEventId;
    }

    protected abstract IntPtr create();

    protected abstract void destroy(IntPtr actionPointer);

    protected virtual void initialize()
	{
		actionRuntime.ExecuteImmediate(getRenderEventID(ActionRenderEventIDS.ACTION_INITIALIZE), ActionPointer);
		Isinitialized = true;
	}

    protected virtual void teardown()
    {
	    Isinitialized = false;
        actionRuntime.ExecuteImmediate(getRenderEventID(ActionRenderEventIDS.ACTION_TEARDOWN), ActionPointer);
    }

    public bool isRegistered()
    {
	    return _renderEventIdOffset > 0;
    }



    protected abstract int getRenderEventIdOffset();

    protected virtual void register()
    {
		actionRuntime.registerAction(this);
		StartCoroutine(WaitUntilActionIsRegisteredEnumerator());
    }

    protected virtual void unregister()
    {
	    _renderEventIdOffset = 0;
        actionRuntime.unregisterAction(this);
    }

    protected void SetupAction()
    {
        if (actionPointer == IntPtr.Zero)
        {
            actionPointer = create();
			register();
			StartCoroutine(InitializeAfterRegistered());
        }
        else
        {
            Debug.LogError("Action already setup");
        }
    }

    public void TeardownAction()
    {
        if (actionPointer != IntPtr.Zero)
        {
            teardown();
            unregister();
            actionPointer = IntPtr.Zero;
        }
        else
        {
            Debug.LogError("TeardownAction not possible");
        }
    }

    protected abstract void ActionRegisterDone();

    protected virtual void Start()
    {
        actionRuntime = ActionRuntime.Instance;
    }
}
