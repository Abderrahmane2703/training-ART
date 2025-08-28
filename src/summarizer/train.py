import art
import asyncio
from dotenv import load_dotenv
import random
#from art.skypilot import SkyPilotBackend
from art.local import LocalBackend

from rollout import rollout, JobOfferScenario
from load_documents import load_documents

load_dotenv()

AGENT_NAME = "job-offer-agent"
PROJECT_NAME = "job-offer-generation"
CLUSTER_NAME = "job-offer-art"


async def main():
    val_contexts, train_contexts = load_documents()

    #backend = await SkyPilotBackend.initialize_cluster(
    #    cluster_name=CLUSTER_NAME,
    #    env_path=".env",
    #    gpu="H100",
    #)

    backend = LocalBackend(
        # set to True if you want your backend to shut down automatically
        # when your client process ends
        in_process=False,
        # local path where the backend will store trajectory logs and model weights
        #path="./.art",
    )


    model = art.TrainableModel(
        name=AGENT_NAME,
        project=PROJECT_NAME,
        base_model="/mnt/c/Users/abder/Documents/CapitaleTech/Training/Summary-RL/gemma-3-270m-it",
    )
    #await backend._experimental_pull_from_s3(model)
    await model.register(backend)

    batch_size = 10  # Process this many documents per batch
    num_epochs = 1  # Number of complete passes through the training data

    start_step = await model.get_step()
    max_steps = 1000
    
    # Tracking for validation-based saving
    best_val_score = 0.0

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        # Shuffle training data at the beginning of each epoch
        random.shuffle(train_contexts)

        # Calculate how many batches we can process in this epoch
        num_batches = min(
            len(train_contexts) // batch_size, (max_steps - start_step) // num_epochs
        )

        for batch in range(num_batches):
            current_step = start_step + epoch * num_batches + batch
            if current_step >= max_steps:
                break

            print(
                f"Epoch {epoch + 1}, Batch {batch + 1}/{num_batches}, Step {current_step}"
            )

            batch_start_idx = batch * batch_size
            batch_end_idx = (batch + 1) * batch_size

            val_groups, train_groups = await asyncio.gather(
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(
                                model,
                                JobOfferScenario(context=context, step=current_step),
                            )
                            for _ in range(2)
                        )
                        for context in val_contexts
                    ),
                    pbar_desc=f"gather val (epoch {epoch + 1})",
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(model, JobOfferScenario(context=context))
                            for _ in range(10)
                        )
                        for context in train_contexts[batch_start_idx:batch_end_idx]
                    ),
                    pbar_desc=f"gather train (epoch {epoch + 1}, batch {batch + 1})",
                ),
            )

            # Calculate validation score (average reward across validation set)
            val_rewards = []
            for group in val_groups:
                for trajectory in group:
                    val_rewards.append(trajectory.reward)
            
            current_val_score = sum(val_rewards) / len(val_rewards) if val_rewards else 0
            
            print(f"Validation score: {current_val_score:.3f} (Best: {best_val_score:.3f})")
            
            await model.log(val_groups)
            await model.delete_checkpoints()
            
            # Train on the batch
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=5e-5),
            )
            
            # Only save to S3 if validation improved
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                print(f"🎉 New best model! Score: {current_val_score:.3f}")
                print(f"Pushing model weights to S3...")
                await backend._experimental_push_to_s3(model)
                print(f"Model weights saved to S3 successfully")
            else:
                print(f"No improvement. Current: {current_val_score:.3f}, Best: {best_val_score:.3f}")
    
    # Training complete summary
    print("\n" + "="*50)
    print("🏁 TRAINING COMPLETE")
    print(f"Final best validation score: {best_val_score:.3f}")
    print(f"Best model saved to S3: job-offer-generation/{AGENT_NAME}")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
