using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JumperRollerAgent : Agent
{

    // Rigidbody rBody;
    void Start()
    {
        // rBody = GetComponent<Rigidbody>();
        lastYposition = this.transform.localPosition.y;
    }

    public Transform Target;
    public Rigidbody rBody;
    private float lastYposition;
    private bool grounded;  

    // When the episode begins
    public override void OnEpisodeBegin(){
        // If the agent fell, reset its velocity
        if(this.transform.localPosition.y < 0){

            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3( 0, 0.5f, 0);

        }

        // Move target to a random spot
        float rand = (int)(Random.value * 4);
        if(rand == 0){
            Target.localPosition = new Vector3(0, 0.5f, 7);
        } else if(rand == 1){
            Target.localPosition = new Vector3(0, 0.5f, -7);
        } else if(rand == 2){
            Target.localPosition = new Vector3(7, 0.5f, 0);
        } else {
            Target.localPosition = new Vector3(-7, 0.5f, 0);
        }
    }


    // What data should I send to the model
    public override void CollectObservations(VectorSensor sensor)
    {
        // Target and Agent positions
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);

        // Agent velocity
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
        sensor.AddObservation(rBody.velocity.y);
    }


    public float forceMultiplier = 20;
    public float jumpPower = 10;
    public float frictionMultiplier = 2;

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

        grounded = (lastYposition == transform.position.y); // Checks if Y has changed since last frame
        lastYposition = transform.position.y;

        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        controlSignal.y = grounded ? actionBuffers.ContinuousActions[2] * jumpPower : 0;
        rBody.AddForce(controlSignal * forceMultiplier);


        Vector3 dragVel = new Vector3(-rBody.velocity.x, 0, -rBody.velocity.z);
        rBody.AddForce(dragVel * frictionMultiplier, ForceMode.Acceleration);

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // Reached target
        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        // Fell off platform
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
        continuousActionsOut[2] = Input.GetAxis("Jump");
    }

}
