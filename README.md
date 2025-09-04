<div align="center">
  <img src="assets/image/zhaiyaotu.png" alt="Header Image" width="100%"/>
  <br/>
  <h1>PPF-framework: Dynamic-Target Potential Pursuit Field Reward for UAV Reinforcement Learning</h1>
</div>

Potential Pursuit Field (PPF),a novel reward shaping framework aimed to address the reward sparsity in reinforcement learning for dynamic target pursuit. By designing a droplet-shaped anisotropic potential field, the proposed PPF model provides dense and direction-aware reward signals while preserving policy invariance through potential-based reward shaping. Building upon PPF, we developed a hierarchical reinforcement learning algorithm, enabling target pursuit and obstacle avoidance in non-line-of-sight(NLOS) environments, simultaneously. 

## Demo
[🎬Bilibili](https://www.bilibili.com/video/BV1nDamziEjh)

[🎬 Watch demo (MP4)](./assets/video/gavideo.mp4)

## PPF-framework

<div align="center">
  <table>
    <tr>
      <td><img src="assets/image/process.png" width="100%"></td>
    </tr>
    <tr>
      <td align="center">PPF-based framework</td>
    </tr>
  </table>
</div>

### A hierarchical reinforcement learning algorithm is proposed based on PPF, which can pursuit the NLOS target under obstacle environment.

## Potential Pursuit Field(PPF)

<p align="center">
  <img src="./assets/image/PPF_simulator.png" alt="Potential Pursuit Field (PPF)" width="480">
  <br>
  <em>Art work of Potential Pursuit Field (PPF)</em>
</p>


### A novel concept of the Potential Pursuit Field (PPF) is proposed to support a continuous and dense reward-shaping function, which can capture anisotropic features and obtain richer gradient information than that of traditional rewards.


## Obstacle-free pursuit
<p align="center">
  <img src="./assets/gif/NO_obs_3.gif" alt="Obstacle-free pursuit" width="49%">
  <img src="./assets/gif/NO_obs_vis_3.gif" alt="Obstacle-free pursuit (visualization)" width="49%">
</p>
<p align="center"></p>


## Obstacle pursuit

<p align="center">
  <img src="./assets/gif/OBS_1.gif" alt="Obstacle pursuit" width="49%">
  <img src="./assets/gif/OBS_1_vis.gif" alt="Obstacle pursuit (visualization)" width="49%">
</p>
<p align="center"><em>Obstacle pursuit (left) and visualization (right)</em></p>





## This is the official for manuscript entitled Dynamic-Target Pursuit Potential Field Reward for UAV Reinforcement Learning submitted to IEEE Transactions on Control Systems Technology

The extire code and corresponding simulation environment will be released later.
