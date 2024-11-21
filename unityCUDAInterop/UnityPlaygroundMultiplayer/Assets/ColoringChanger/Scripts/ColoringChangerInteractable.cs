using B3D;
using MixedReality.Toolkit;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Interaction.Toolkit;

[RequireComponent(typeof(MeshRenderer))]
public class ColoringChangerInteractable : MRTKBaseInteractable
{
	// Used draw a full line between current frame + last frame's "paintbrush" position.
	private Dictionary<UnityEngine.XR.Interaction.Toolkit.Interactors.IXRInteractor, Vector2> lastPositions = new Dictionary<UnityEngine.XR.Interaction.Toolkit.Interactors.IXRInteractor, Vector2>();

	public delegate void StartEndPositionHandler(Vector2 start, Vector2 end);
	public event StartEndPositionHandler StartEndPositionEvent;

	public override void ProcessInteractable(XRInteractionUpdateOrder.UpdatePhase updatePhase)
	{
		// Dynamic is effectively just your normal Update().
		if (updatePhase == XRInteractionUpdateOrder.UpdatePhase.Dynamic)
		{
			foreach (var interactor in interactorsSelecting)
			{
				// attachTransform will be the actual point of the touch interaction (e.g. index tip)
				// Most applications will probably just end up using this local touch position.
				Vector3 localTouchPosition = transform.InverseTransformPoint(interactor.GetAttachTransform(this).position);

				// For whiteboard drawing: compute UV coordinates on texture by flattening Vector3 against the plane and adding 0.5f.
				Vector2 uvTouchPosition = new Vector2(localTouchPosition.x + 0.5f, localTouchPosition.y + 0.5f);

				Vector2 clampedUvTouchPosition = Vector2.Max(Vector2.zero, Vector2.Min(uvTouchPosition, Vector2.one));

				// Have we seen this interactor before? If not, last position = current position.
				if (!lastPositions.TryGetValue(interactor, out Vector2 lastPosition))
				{
					clampedUvTouchPosition.x = Mathf.Max(float.Epsilon, clampedUvTouchPosition.x);
					lastPosition = clampedUvTouchPosition;
				}

				StartEndPositionEvent?.Invoke(lastPosition, clampedUvTouchPosition);
				

				// Write/update the last-position.
				if (lastPositions.ContainsKey(interactor))
				{
					lastPositions[interactor] = clampedUvTouchPosition;
				}
				else
				{
					lastPositions.Add(interactor, clampedUvTouchPosition);
				}
			}

		}
	}

	/// <inheritdoc />
	protected override void OnSelectExited(SelectExitEventArgs args)
	{
		base.OnSelectExited(args);
		StartEndPositionEvent?.Invoke(new(1.1f, 0), new(1.1f, 0));
		// Remove the interactor from our last-position collection when it leaves.
		lastPositions.Remove(args.interactorObject);
	}
}
