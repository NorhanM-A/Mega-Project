# Mega-Project
## Task 1: Simulating a Flight using Mission Planner.
Google Drive link: https://drive.google.com/drive/folders/1TBIbz1A3e-m7vlDX9Z_a0Pfv6xAPN0ae?usp=sharing

# Mission Planner Flight Simulation README

This README provides an overview of how to simulate a flight using Mission Planner and includes three videos showcasing different flight scenarios.



## Introduction

Mission Planner is a powerful ground control station software for planning, monitoring, and simulating drone flights. This README presents three video demonstrations showcasing different aspects of flight simulation using Mission Planner.

## Video 1: Simple Simulated Flight

[![Simple Simulated Flight] https://drive.google.com/file/d/1W6iULGM8bOEoGTxtCGJqHcC4O0JFpD3J/view?usp=sharing
- **Description:** This video demonstrates a basic simulated flight using Mission Planner. The drone follows a predefined flight path with no obstacles.

## Video 2: Smart RTL (Return to Launch)

[![Smart RTL]https://drive.google.com/file/d/1PxXEfLaikQFgcYcLYuM41NAcjdTTrjM3/view?usp=drive_link
- **Description:** In this video, we showcase the Smart RTL feature in Mission Planner. Smart RTL allows the drone to automatically return to its launch point if it encounters any issues or loses communication with the ground station.

## Video 3: Obstacle Avoidance

[![Obstacle Avoidance]https://drive.google.com/file/d/1dGji04lGaqQHL1SUb-QbIlr27KNVm8FN/view?usp=drive_link
- **Description:** This video illustrates obstacle avoidance in a simulated flight. When the drone detects an obstacle along its flight path, it autonomously returns to the home location to avoid a collision.
## Video 4: Obstacle Avoidance (Successfully avoiding the obstacle without RTL)
https://drive.google.com/file/d/1Hdw9mM34UobvORVkSGMm4uK9hfifm69J/view?usp=sharing
- **Description:** In this video successful obstacle avoidance was achieved by going around the obstacle with no need for returning to the launch point and also without cancelling other waypoints along gthe way.
## Custom Programming for Obstacle Avoidance

It's important to note that the obstacle avoidance behavior demonstrated in Video 3, where the drone returns home or avoids the remaining waypoints upon detecting an obstacle, is a basic behavior provided by default in many flight planning tools, including Mission Planner. If you require more advanced obstacle avoidance capabilities, such as having the drone navigate around obstacles while continuing its mission, custom programming and additional hardware may be necessary.

Creating a drone that can intelligently navigate around obstacles in real-time typically involves:

- **Custom Software Development:** You'll need to develop or integrate custom software that enables the drone to perceive and identify obstacles using sensors (LiDAR, depth cameras) and make real-time decisions to adjust its flight path.

- **Advanced Flight Controller:** Some flight controllers offer advanced obstacle avoidance features, but you may need to configure and fine-tune them to suit your specific needs.

- **Sensor Integration:** Depending on the complexity of your project, you may need to integrate additional sensors and hardware to enhance obstacle detection and avoidance.

- **Testing and Calibration:** Rigorous testing and calibration of your obstacle avoidance system will be essential to ensure reliable performance in real-world scenarios.



