using MixedReality.Toolkit;
using MixedReality.Toolkit.SpatialManipulation;
using UnityEngine;
using static UnityEngine.Rendering.DebugUI.Table;

public class RotationSolver : MixedReality.Toolkit.SpatialManipulation.Solver
{
	public Vector3 localRotationOffset = Vector3.zero;

	private Quaternion SolverReferenceRotation => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.rotation : Quaternion.identity;

	private Vector3 SolverReferenceDirection => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.forward : Vector3.forward;
	private Vector3 SolverReferencePosition => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.position : transform.position;

	private Vector3 UpReference => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.up : Vector3.up;

	private Vector3 ReferencePoint => transform.position;

	private float CurrentDistance => Vector3.Distance(SolverReferencePosition, ReferencePoint);

	public override void SolverUpdate()
	{
		// Smoothing does not work with radial solvers.
		
		Smoothing = false;

		// Move Lerp time is not considered
		Vector3 newforwardDirection = Vector3.RotateTowards(Vector3.Normalize(ReferencePoint - SolverReferencePosition), SolverReferenceDirection, SolverHandler.DeltaTime * 3 , 0.0f);
		GoalPosition = SolverReferencePosition + newforwardDirection * CurrentDistance;
		Quaternion newRot = SolverReferenceRotation;
		if(SolverHandler.TransformTarget != null)
		{ 
			Quaternion oldRot = SolverReferenceRotation;
			SolverHandler.TransformTarget.localRotation = Quaternion.Euler(localRotationOffset);
			newRot = SolverReferenceRotation;
			SolverHandler.TransformTarget.rotation = oldRot;
		}
		GoalRotation = SmoothTo(transform.rotation, newRot, SolverHandler.DeltaTime, RotateLerpTime);
		// Add local rotation offset to 
	}

	public void viewXY()
	{
		localRotationOffset = new Vector3(0, 0, 0);
		enabled = false;
		enabled = true;
	}

	public void viewXZ()
	{
		localRotationOffset = new Vector3(90, 0, 0);
		enabled = false;
		enabled = true;
	}

	public void viewZY()
	{
		localRotationOffset = new Vector3(0, 90, 0);
		enabled = false;
		enabled = true;
	}

	public void resetScale()
	{
		transform.localScale = Vector3.one;
	}

	public void resetToOrigin()
	{
		enabled = false;
		transform.localScale = Vector3.one;
		transform.localRotation = Quaternion.identity;
		transform.localPosition = Vector3.zero;

	}
}
