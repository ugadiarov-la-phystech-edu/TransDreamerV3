import sys
from pathlib import Path

# Add dynalang to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamerv3.embodied.envs.homegrid import HomeGrid
from PIL import Image


def main():
    # Create HomeGrid environment
    env = HomeGrid(
        task="task",
        size=(64, 64),
        max_steps=50,
        num_trashobjs=2,
        num_trashcans=2,
        p_teleport=0.05,
        p_unsafe=0.0,
        vis=True  # Enable text overlay
    )
    
    print("Created HomeGrid environment")
    print(f"Action space: {env.action_space}")
    
    # Reset and run episode
    obs = env.reset()
    print(obs)
    
    frames = []
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 50:
        # Take random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Save frame with text overlay
        if 'log_image' in obs:
            frames.append(obs['log_image'])
        
        step += 1
        if step % 10 == 0:
            print(f"Step {step}, Total reward: {total_reward:.2f}")
    
    print(f"\nFinished! Steps: {step}, Total reward: {total_reward:.2f}")
    
    # Save some frames as images
    if frames:
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            "homegrid_episode.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,  # 100ms per frame
            loop=0
        )
        print(f"Saved {len(frames)} frames as homegrid_episode.gif")
        
        # Also save first, middle, and last frame as PNG
        Image.fromarray(frames[0]).save("homegrid_frame_start.png")
        Image.fromarray(frames[len(frames)//2]).save("homegrid_frame_middle.png")
        Image.fromarray(frames[-1]).save("homegrid_frame_end.png")
        print(f"Saved 3 frames as PNG files")
        # Save first, middle, and last frame
        Image.fromarray(frames[0]).save("homegrid_frame_start.png")
        Image.fromarray(frames[len(frames)//2]).save("homegrid_frame_middle.png")
        Image.fromarray(frames[-1]).save("homegrid_frame_end.png")
        print(f"Saved 3 frames as PNG files")


if __name__ == "__main__":
    main()