![image](https://user-images.githubusercontent.com/101176694/226290594-60e0bbcc-13ac-48ae-a04f-6c3c0935d48b.png)

## Objective
Making TurtleBot3 run toward its goal without colliding with obstacles, using only reinforcement learning
![image](https://user-images.githubusercontent.com/101176694/226274046-596f9b72-9c73-4e59-aeb9-052a7e90db78.png)

## Turtlebot3 hardware
![image](https://user-images.githubusercontent.com/101176694/226275760-ed28d3cb-781a-480c-b628-85f20565ac28.png)

## Implementation
- We created a custom environment, using GYM(OpenAI) to connect with ROS2 Gazebo, which's robot operating system of TurtleBot3.
- trained robot by Proximal Policy Optimization(PPO) model from Stable Baseline3.

![image](https://user-images.githubusercontent.com/101176694/226275888-59db9360-7304-4cef-b161-c65401bb7b0e.png)

## Experiment
- visualized result via TensorBoard and tuned model by changing/calculating observation (state in reinforcement learning) and reward functions.
- example on slide page 14 (video)
![image](https://user-images.githubusercontent.com/101176694/226275942-c8c13e00-23c4-4db2-92df-cfe4e323a4f2.png)



## slide
https://www.canva.com/design/DAFbTrYqlBE/tZxOKyIbGi-EwieWo8HZXA/view
