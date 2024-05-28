using MixedReality.Toolkit.Input;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.XR.Interaction.Toolkit;
using static MixedReality.Toolkit.Input.XRRayInteractorExtensions;

public class ImageManipulationPositionProvider : MonoBehaviour
{
    private XRRayInteractor interactor;

    private TargetHitDetails selectedHitDetails = new TargetHitDetails();
    private RectTransform parentRect = null;

    private void Update()
    {
        Vector2 la;
        if(GetPosition(out la))
        {
            Debug.Log(la);
        }
    }

    public bool GetPosition(out Vector2 currentPosition)
    {
        if(interactor == null )
        {
            currentPosition = Vector2.negativeInfinity;
            return false;
        }
        
        if(parentRect != null)
        {
            currentPosition = PointToNormalizedUnclamped(parentRect.rect, selectedHitDetails.TargetLocalHitPoint + selectedHitDetails.HitTargetTransform.localPosition);
        }
        else
        {
            currentPosition = selectedHitDetails.TargetLocalHitPoint + selectedHitDetails.HitTargetTransform.localPosition;
        }
        return true;
    }

    public void StartPointAtImage(SelectEnterEventArgs args)
    {
        if (interactor == null)
        {
            interactor = args.interactorObject as XRRayInteractor;
            interactor.TryLocateTargetHitPoint(args.interactableObject, out selectedHitDetails);
            parentRect = selectedHitDetails.HitTargetTransform.parent as RectTransform;
        }  
    }

    public void EndPointAtImage(SelectExitEventArgs args)
    {
        interactor = null;
        parentRect = null;
    }

    public static float InverseLerpUnclamped(float a, float b, float value)
    {
        if (a != b)
        {
            return (value - a) / (b - a);
        }

        return 0f;
    }

    public static Vector2 PointToNormalizedUnclamped(Rect rectangle, Vector2 point)
    {
        return new Vector2(InverseLerpUnclamped(rectangle.x, rectangle.xMax, point.x), InverseLerpUnclamped(rectangle.y, rectangle.yMax, point.y));
    }
}
